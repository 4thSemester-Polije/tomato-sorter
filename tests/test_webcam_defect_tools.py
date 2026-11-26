import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from controller import ml_client
from controller.ml_client import build_multipart, check_api, make_image_predictor
from ml import api
from ml.metrics import wilson_interval
from tools import eval_webcam_defect as ev
from tools.camera_predict import (
    LOG_COLUMNS,
    analyze,
    append_result_row,
    next_row_number,
    result_row,
)

BG = (235, 235, 235)
RESULT = {
    "architecture": "resnext101",
    "grade": "A",
    "ripeness": {"class": "matang", "level": 1, "confidence": 0.9, "probabilities": {}},
    "defect": {"class": "reject", "level": 3, "confidence": 0.8, "p_reject": 0.8,
               "threshold": 0.5, "input": "crop_neutral", "probabilities": {}},
}


def _frame(with_tomato):
    img = np.full((480, 640, 3), BG, dtype=np.uint8)
    if with_tomato:
        cv2.circle(img, (320, 240), 90, (30, 30, 200), -1)
    return img


# ------------------------------------------------------------------ klien multipart


def test_multipart_is_byte_identical_to_before_when_no_frame_is_attached():
    body, content_type = build_multipart(b"JPEG")
    boundary = content_type.split("boundary=")[1]
    expected = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="frame.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode() + b"JPEG" + f"\r\n--{boundary}--\r\n".encode()

    assert body == expected


def test_multipart_with_frame_is_parsed_by_the_real_api(monkeypatch):
    class Model:
        def __init__(self, probabilities):
            self.probabilities = np.array([probabilities], dtype=np.float32)

        def predict(self, batch, verbose=0):
            return self.probabilities

    meta = {"class_names": ["healthy", "reject"], "data_preparation": {"background": "neutral"}}
    monkeypatch.setattr(
        api, "_load_models",
        lambda models_dir, architecture="resnext101": (
            Model([0.9, 0.05, 0.05]), Model([0.1, 0.9]),
            {"class_names": ["matang", "setengah_matang", "mentah"]}, meta, lambda x: x / 255.0,
        ),
    )
    monkeypatch.setenv("TOMATO_GRADE_USES_DEFECT", "1")
    scene = Image.new("RGB", (640, 480), (255, 255, 255))
    ImageDraw.Draw(scene).ellipse((220, 140, 420, 340), fill=(200, 40, 30))
    buffer = io.BytesIO()
    scene.save(buffer, format="JPEG")
    body, content_type = build_multipart(buffer.getvalue(), frame_jpeg=buffer.getvalue())

    response = TestClient(api.app).post("/predict", content=body, headers={"Content-Type": content_type})

    assert response.status_code == 200
    assert response.json()["defect"]["p_reject"] == pytest.approx(0.9, abs=1e-6)


def test_image_predictor_declares_that_it_accepts_a_raw_frame():
    assert make_image_predictor("http://127.0.0.1:9000").accepts_frame is True


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_check_api_shows_which_defect_model_is_active(monkeypatch):
    health = {"status": "ready", "architecture": "resnext101",
              "architectures": {"resnext101": {"status": "ready", "missing_files": []}},
              "defect_head": {"trained_input": "crop_neutral", "threshold": 0.5, "used_for_grade": False}}
    monkeypatch.setattr(ml_client.request, "urlopen", lambda url, timeout=0: _Response(health))

    ok, message = check_api("http://x")

    assert ok and "crop_neutral" in message and "ambang 0.5" in message and "ikut grade: tidak" in message


def test_check_api_message_is_unchanged_for_an_older_api(monkeypatch):
    health = {"status": "ready", "architecture": "resnext101",
              "architectures": {"resnext101": {"status": "ready", "missing_files": []}}}
    monkeypatch.setattr(ml_client.request, "urlopen", lambda url, timeout=0: _Response(health))

    assert check_api("http://x") == (True, "API model siap (resnext101)")


# ------------------------------------------------------------------ camera_predict


def test_analyze_sends_the_raw_frame_only_to_predictors_that_accept_it():
    seen: dict = {}

    def modern(jpeg, frame_jpeg=None):
        seen.update(enhanced=len(jpeg), frame=len(frame_jpeg) if frame_jpeg else None)
        return RESULT

    modern.accepts_frame = True
    legacy_calls = []

    def legacy(jpeg):
        legacy_calls.append(len(jpeg))
        return RESULT

    assert analyze(_frame(True), _frame(False), None, modern)["valid"]
    assert seen["enhanced"] > 0 and seen["frame"] > 0
    assert analyze(_frame(True), _frame(False), None, legacy)["valid"] and len(legacy_calls) == 1


def test_result_row_for_an_assessed_defect():
    row = result_row(3, 1700000000000, "raw_1.jpg", {"result": RESULT})

    assert row["no"] == 3 and row["grade"] == "A" and row["defect_status"] == "dinilai"
    assert row["defect_p_reject"] == 0.8 and row["defect_input"] == "crop_neutral"
    assert row["true_defect_level"] == "" and set(row) == set(LOG_COLUMNS)


def test_result_row_reads_p_reject_from_the_hidden_raw_defect():
    hidden = {**RESULT, "defect": {"class": "tidak_dinilai", "level": None, "reason": "tidak_ikut_grade"},
              "defect_raw": RESULT["defect"]}

    row = result_row(1, 1, "r.jpg", {"result": hidden})

    assert row["defect_class"] == "tidak_dinilai" and row["defect_status"] == "dinilai"
    assert row["defect_p_reject"] == 0.8


def test_result_row_records_why_the_defect_was_not_assessed_and_invalid_captures():
    skipped = {**RESULT, "defect": {"class": "tidak_dinilai", "level": None, "reason": "crop_gagal"}}

    assert result_row(1, 1, "r.jpg", {"result": skipped})["defect_status"] == "crop_gagal"
    invalid = result_row(2, 2, "r2.jpg", {"result": None, "valid": False})
    assert invalid["defect_status"] == "tomat_tidak_valid" and invalid["grade"] == ""


def test_result_row_falls_back_to_probabilities_for_a_legacy_model():
    legacy = {**RESULT, "defect": {"class": "reject", "level": 3, "confidence": 0.7,
                                   "probabilities": {"healthy": 0.3, "reject": 0.7}}}

    assert result_row(1, 1, "r.jpg", {"result": legacy})["defect_p_reject"] == 0.7


def test_log_csv_has_one_header_and_continues_numbering(tmp_path: Path):
    path = tmp_path / "log" / "results.csv"
    assert next_row_number(path) == 1
    append_result_row(path, result_row(1, 10, "a.jpg", {"result": RESULT}))
    assert next_row_number(path) == 2
    append_result_row(path, result_row(2, 20, "b.jpg", {"result": RESULT}))

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [row["no"] for row in rows] == ["1", "2"] and "\r" not in path.read_text(encoding="utf-8")


# ------------------------------------------------------------------ evaluasi


def _rows(level3, level1, level2=()):
    """Setiap elemen: p_reject (angka) atau None bila tidak dinilai."""
    rows = []
    for level, scores in ((3, level3), (1, level1), (2, level2)):
        for score in scores:
            rows.append({"true_defect_level": str(level), "defect_p_reject": "" if score is None else str(score)})
    rows.append({"true_defect_level": "", "defect_p_reject": "0.99"})  # belum berlabel: diabaikan
    return rows


def test_wilson_interval():
    low, high = wilson_interval(8, 9)
    assert low == pytest.approx(0.565, abs=0.01) and high == pytest.approx(0.980, abs=0.01)
    assert wilson_interval(0, 0) is None
    assert wilson_interval(0, 10)[0] == 0.0


def test_evaluation_passes_with_enough_positives_good_recall_and_low_false_positives():
    result = ev.evaluate(_rows([0.9] * 8 + [0.3, 0.2], [0.1] * 19 + [0.8]), threshold=0.5)

    assert result["rows_labeled"] == 30 and result["recall"] == 0.8 and result["fpr"] == 0.05
    assert result["verdict"] == "LULUS"


def test_not_assessed_photos_count_as_missed_defects():
    result = ev.evaluate(_rows([0.9] * 6 + [None] * 4, [0.1] * 20), threshold=0.5)

    assert result["per_level"][3]["not_assessed"] == 4
    assert result["recall"] == 0.6 and result["verdict"] == "BELUM LULUS"


def test_too_many_false_positives_fail_the_gate():
    result = ev.evaluate(_rows([0.9] * 10, [0.1] * 16 + [0.9] * 4), threshold=0.5)

    assert result["fpr"] == 0.2 and result["verdict"] == "BELUM LULUS"


def test_gate_is_not_decided_with_fewer_than_ten_defective_tomatoes():
    result = ev.evaluate(_rows([0.9] * 9, [0.1] * 30), threshold=0.5)

    assert result["verdict"].startswith("TIDAK DAPAT DINILAI")


def test_the_frozen_threshold_is_what_decides_the_verdict_not_the_sweep():
    rows = _rows([0.45] * 10, [0.1] * 20)

    assert ev.evaluate(rows, threshold=0.5)["verdict"] == "BELUM LULUS"
    assert ev.evaluate(rows, threshold=0.4)["verdict"] == "LULUS"


def test_render_and_cli(tmp_path: Path, monkeypatch, capsys):
    path = tmp_path / "results.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["true_defect_level", "defect_p_reject"])
        writer.writeheader()
        writer.writerows(_rows([0.9] * 10, [0.1] * 20, [0.2] * 5))
    out = tmp_path / "laporan.md"
    monkeypatch.setattr(sys, "argv", ["eval", "--csv", str(path), "--out", str(out)])

    ev.main()

    text = capsys.readouterr().out
    assert "**LULUS**" in text and "Sapuan ambang" in text and "95%" in text
    assert out.read_text(encoding="utf-8").startswith("# Evaluasi uji webcam")


def test_cli_refuses_a_csv_without_labels(tmp_path: Path, monkeypatch):
    path = tmp_path / "results.csv"
    path.write_text("true_defect_level,defect_p_reject\n,0.5\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["eval", "--csv", str(path)])

    with pytest.raises(SystemExit, match="true_defect_level"):
        ev.main()
