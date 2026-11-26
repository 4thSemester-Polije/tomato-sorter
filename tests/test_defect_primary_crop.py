import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

import ml.train as train_module


class _FakeModel:
    def __init__(self, p_reject: float) -> None:
        self.p_reject = p_reject
        self.shapes: list[tuple[int, ...]] = []

    def predict(self, batch, verbose=0):
        self.shapes.append(tuple(batch.shape))
        return np.array([[1.0 - self.p_reject, self.p_reject]])


def _photo(path: Path, tomato: bool = True, seed: int = 0) -> Path:
    image = np.full((300, 500, 3), 255, dtype=np.uint8)
    if tomato:
        rng = np.random.default_rng(seed)
        cv2.ellipse(image, (250, 150), (60, 66), 0, 0, 360, (30, 40, 200), -1)
        cv2.circle(image, (230 + int(rng.integers(0, 40)), 140), 10, (10, 10, 10), -1)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    path.write_bytes(encoded.tobytes())
    return path


@pytest.fixture
def primary(tmp_path: Path, monkeypatch):
    levels = [1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 1]  # foto terakhir kosong (tanpa tomat)
    samples = []
    for index, level in enumerate(levels):
        path = _photo(tmp_path / f"T{index:03d}.jpg", tomato=index != len(levels) - 1, seed=index)
        samples.append((path, {"tingkat_cacat": level}))
    monkeypatch.setattr(train_module, "_primary_test_samples", lambda root: samples)
    monkeypatch.setattr(
        train_module, "get_preprocess_input", lambda architecture: (lambda pixels: pixels / 255.0)
    )
    return samples


def test_full_frame_mode_scores_every_photo(primary, tmp_path: Path):
    model = _FakeModel(0.9)

    result = train_module._evaluate_defect_primary(tmp_path, model, ["healthy", "reject"])

    assert result["input"] == "full_frame"
    assert result["scored_images"] == len(primary) and result["no_crop"] == 0
    assert {row["crop_status"] for row in result["rows"]} == {"full_frame"}
    assert set(model.shapes) == {(1, 224, 224, 3)}


def test_crop_mode_skips_photos_without_a_tomato_and_reports_them(primary, tmp_path: Path):
    model = _FakeModel(0.9)

    result = train_module._evaluate_defect_primary(
        tmp_path, model, ["healthy", "reject"], crop_mode="neutral"
    )

    assert result["input"] == "crop_neutral"
    assert result["no_crop"] == 1 and result["scored_images"] == len(primary) - 1
    failed = [row for row in result["rows"] if row["crop_status"] == "gagal"]
    assert len(failed) == 1 and failed[0]["p_reject"] is None
    assert result["per_level"][1]["images"] == 6  # 7 foto level 1, satu tanpa crop
    assert result["per_level"][1]["no_crop"] == 1
    assert result["level1_vs_level3"]["negatives"] == 6
    assert set(model.shapes) == {(1, 224, 224, 3)}  # crop 256 diubah ke ukuran masukan model


def test_crop_status_is_written_to_the_predictions_rows(primary, tmp_path: Path):
    result = train_module._evaluate_defect_primary(
        tmp_path, _FakeModel(0.2), ["healthy", "reject"], crop_mode="keep"
    )

    statuses = {row["crop_status"] for row in result["rows"]}
    assert statuses == {"crop_keep", "gagal"}
    sections = dict(train_module._defect_primary_sections(result))
    assert "tanpa crop" in sections["Primer - model cacat per level cacat sebenarnya"]
    assert "crop_keep" in sections["Primer - ringkasan model cacat"]


def test_resolve_primary_crop_auto_follows_the_dataset_preparation():
    assert train_module._resolve_primary_crop("auto", {"background": "neutral"}) == "neutral"
    assert train_module._resolve_primary_crop("auto", {"background": "keep"}) == "keep"
    assert train_module._resolve_primary_crop("auto", None) is None


def test_resolve_primary_crop_warns_when_it_differs_from_the_dataset(caplog):
    with caplog.at_level(logging.WARNING, logger="ml.train"):
        resolved = train_module._resolve_primary_crop(None, {"background": "neutral"})

    assert resolved is None
    assert "--primary-crop neutral" in caplog.text


def test_resolve_primary_crop_is_quiet_when_consistent(caplog):
    with caplog.at_level(logging.WARNING, logger="ml.train"):
        assert train_module._resolve_primary_crop("neutral", {"background": "neutral"}) == "neutral"
        assert train_module._resolve_primary_crop(None, None) is None

    assert caplog.text == ""


def test_read_preparation_info(tmp_path: Path):
    assert train_module._read_preparation_info(None) is None
    assert train_module._read_preparation_info(tmp_path) is None
    (tmp_path / "prepare_report.json").write_text("{rusak", encoding="utf-8")
    assert train_module._read_preparation_info(tmp_path) is None
    (tmp_path / "prepare_report.json").write_text(
        json.dumps({"parameters": {"background": "neutral", "size": 256}}), encoding="utf-8"
    )
    assert train_module._read_preparation_info(tmp_path) == {"background": "neutral", "size": 256}


def test_train_rejects_unknown_primary_crop(tmp_path: Path):
    with pytest.raises(ValueError, match="primary_crop"):
        train_module.train(tmp_path, tmp_path / "m", primary_crop="blur")
