import io
import json
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from ml import api
from vision.crop import crop_for_inference


class _Model:
    def __init__(self, probabilities):
        self.probabilities = np.array([probabilities], dtype=np.float32)
        self.calls: list[tuple[int, ...]] = []

    def predict(self, batch, verbose=0):
        self.calls.append(tuple(batch.shape))
        return self.probabilities


CROP_META = {"class_names": ["healthy", "reject"], "data_preparation": {"background": "neutral"}}
LEGACY_META = {"class_names": ["healthy", "reject"]}
RIPENESS_META = {"class_names": ["matang", "setengah_matang", "mentah"]}


def _scene(tomato: bool = True) -> Image.Image:
    image = Image.new("RGB", (640, 480), (255, 255, 255))
    if tomato:
        ImageDraw.Draw(image).ellipse((220, 140, 420, 340), fill=(200, 40, 30))
    return image


def _jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def models(monkeypatch, tmp_path):
    """Pasang model palsu; kembalikan (ripeness_model, defect_model). Ripeness bawaan: matang."""

    def install(p_reject: float, metadata=CROP_META, ripeness=(0.9, 0.05, 0.05)):
        ripeness_model, defect_model = _Model(ripeness), _Model([1.0 - p_reject, p_reject])
        monkeypatch.setattr(
            api,
            "_load_models",
            lambda models_dir, architecture="resnext101": (
                ripeness_model, defect_model, RIPENESS_META, metadata, lambda pixels: pixels / 255.0,
            ),
        )
        return ripeness_model, defect_model

    monkeypatch.delenv("TOMATO_DEFECT_THRESHOLD", raising=False)
    monkeypatch.delenv("TOMATO_GRADE_USES_DEFECT", raising=False)
    return install


def _predict(source, tmp_path):
    return api.predict_image(_scene(), tmp_path, "resnext101", defect_source=source)


def test_crop_model_scores_the_crop_of_the_raw_frame(models, tmp_path):
    _, defect_model = models(0.8)

    defect = _predict(_scene(), tmp_path)["defect"]

    assert defect["class"] == "reject" and defect["level"] == 3
    assert defect["p_reject"] == pytest.approx(0.8) and defect["confidence"] == pytest.approx(0.8)
    assert defect["input"] == "crop_neutral" and defect["threshold"] == 0.5
    assert defect_model.calls == [(1, 224, 224, 3)]


def test_threshold_comes_from_the_environment(models, tmp_path, monkeypatch):
    models(0.8)
    monkeypatch.setenv("TOMATO_DEFECT_THRESHOLD", "0.9")

    defect = _predict(_scene(), tmp_path)["defect"]

    assert defect["class"] == "healthy" and defect["level"] == 1
    assert defect["confidence"] == pytest.approx(0.2) and defect["threshold"] == 0.9


@pytest.mark.parametrize("bad", ["abc", "0", "1", "1.5", "-0.2"])
def test_invalid_threshold_falls_back_to_default_with_a_warning(bad, monkeypatch, caplog):
    monkeypatch.setenv("TOMATO_DEFECT_THRESHOLD", bad)

    with caplog.at_level(logging.WARNING, logger="uvicorn.error"):
        assert api._defect_threshold() == 0.5

    assert "TOMATO_DEFECT_THRESHOLD" in caplog.text


def test_without_the_raw_frame_a_crop_model_does_not_guess(models, tmp_path):
    _, defect_model = models(0.99)

    result = api.predict_image(_scene(), tmp_path, "resnext101")

    assert result["defect"]["class"] == "tidak_dinilai"
    assert result["defect"]["reason"] == "tanpa_frame_mentah"
    assert defect_model.calls == []


def test_a_frame_without_a_tomato_is_not_assessed(models, tmp_path):
    models(0.99)

    defect = _predict(_scene(tomato=False), tmp_path)["defect"]

    assert defect["class"] == "tidak_dinilai" and defect["reason"] == "crop_gagal"


def test_legacy_full_frame_model_keeps_the_old_behaviour(models, tmp_path):
    _, defect_model = models(0.8, metadata=LEGACY_META)

    defect = _predict(_scene(), tmp_path)["defect"]

    assert defect["class"] == "reject" and "p_reject" not in defect
    assert defect_model.calls == [(1, 224, 224, 3)]


def test_grade_ignores_defect_by_default(models, tmp_path):
    models(0.95)

    result = _predict(_scene(), tmp_path)

    assert result["grade"] == "A" and result["grade_basis"] == "ripeness"


def test_enabled_defect_makes_a_clear_defect_grade_c(models, tmp_path, monkeypatch):
    models(0.95)
    monkeypatch.setenv("TOMATO_GRADE_USES_DEFECT", "1")

    result = _predict(_scene(), tmp_path)

    assert result["grade"] == "C" and result["grade_basis"] == "ripeness+defect"


def test_enabled_defect_that_could_not_be_assessed_falls_back_to_ripeness(models, tmp_path, monkeypatch):
    models(0.95)
    monkeypatch.setenv("TOMATO_GRADE_USES_DEFECT", "1")

    result = api.predict_image(_scene(), tmp_path, "resnext101")  # tanpa frame mentah

    assert result["grade"] == "A" and result["grade_basis"] == "ripeness"


def test_public_result_explains_why_the_defect_is_hidden():
    assessed = {"grade_basis": "ripeness", "defect": {"level": 3, "p_reject": 0.9, "class": "reject"}}
    skipped = {"grade_basis": "ripeness", "defect": {"level": None, "reason": "crop_gagal"}}

    shown_assessed = api._public_result(assessed)
    shown_skipped = api._public_result(skipped)

    assert shown_assessed["defect"]["reason"] == "tidak_ikut_grade"
    assert shown_assessed["defect_raw"]["p_reject"] == 0.9
    assert shown_skipped["defect"]["reason"] == "crop_gagal"


def _post(client, **extra):
    files = {"file": ("e.jpg", _jpeg(_scene()), "image/jpeg")}
    if extra.pop("with_frame", False):
        files["frame"] = ("f.jpg", _jpeg(_scene()), "image/jpeg")
    return client.post("/predict", files=files, params=extra).json()


def test_endpoint_uses_the_uploaded_frame_for_the_defect_head(models, monkeypatch):
    models(0.9)
    monkeypatch.setenv("TOMATO_GRADE_USES_DEFECT", "1")

    body = _post(TestClient(api.app), with_frame=True)

    assert body["grade"] == "C" and body["defect"]["class"] == "reject"
    assert body["defect"]["p_reject"] == pytest.approx(0.9, abs=1e-6)


def test_endpoint_hides_the_defect_but_keeps_p_reject_when_not_used_for_grade(models):
    models(0.9)

    body = _post(TestClient(api.app), with_frame=True)

    assert body["grade"] == "A"
    assert body["defect"]["class"] == "tidak_dinilai" and body["defect"]["reason"] == "tidak_ikut_grade"
    assert body["defect_raw"]["p_reject"] == pytest.approx(0.9, abs=1e-6)


def test_endpoint_without_frame_reports_missing_raw_frame(models, monkeypatch):
    models(0.9)
    monkeypatch.setenv("TOMATO_GRADE_USES_DEFECT", "1")

    body = _post(TestClient(api.app))

    assert body["grade"] == "A" and body["defect_raw"]["reason"] == "tanpa_frame_mentah"


def test_apply_pcv_uses_the_original_image_for_the_defect_head(models, monkeypatch):
    _, defect_model = models(0.9)
    monkeypatch.setenv("TOMATO_GRADE_USES_DEFECT", "1")
    fake = types.ModuleType("vision.features")
    fake.extract_pcv_features = lambda frame, crop_size=300: {
        "img_enhanced_rgb": np.full((300, 300, 3), 90, dtype=np.uint8),  # tanpa tomat sama sekali
        "persen_merah": 1.0, "persen_oranye": 0.0, "persen_kuning": 0.0, "persen_hijau": 0.0,
        "persentase_cacat": 0.0, "jumlah_kontur_cacat": 0, "luas_piksel": 1.0,
    }
    monkeypatch.setitem(sys.modules, "vision.features", fake)

    body = _post(TestClient(api.app), apply_pcv="true")

    assert body["defect"]["class"] == "reject"  # crop berhasil dari gambar asli, bukan dari gambar PCV abu-abu
    assert "pcv" in body and len(defect_model.calls) == 1


def test_health_reports_which_defect_model_is_used(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "DEFAULT_MODELS_DIR", tmp_path)
    (tmp_path / "defect_resnext101.keras").write_bytes(b"x" * 10)
    (tmp_path / "defect_resnext101.json").write_text(json.dumps(CROP_META), encoding="utf-8")
    monkeypatch.setenv("TOMATO_DEFECT_THRESHOLD", "0.4")

    head = TestClient(api.app).get("/health").json()["defect_head"]

    assert head["trained_input"] == "crop_neutral" and head["threshold"] == 0.4
    assert head["model_exists"] is True and head["used_for_grade"] is False
    assert head["data_preparation"] == {"background": "neutral"}


def test_health_marks_a_legacy_defect_model_as_full_frame(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "DEFAULT_MODELS_DIR", tmp_path)
    (tmp_path / "defect_resnext101.json").write_text(json.dumps(LEGACY_META), encoding="utf-8")

    assert TestClient(api.app).get("/health").json()["defect_head"]["trained_input"] == "full_frame"


def test_warm_up_also_runs_the_defect_head_of_a_crop_model(models, tmp_path):
    _, defect_model = models(0.5)

    assert api.warm_up(tmp_path, "resnext101") is True
    assert len(defect_model.calls) == 1  # hanya lewat adegan sintetis; foto kosong tidak memanggilnya


def test_crop_for_inference_returns_a_square_pil_crop_or_none():
    crop = crop_for_inference(_scene(), "neutral", size=128)

    assert crop is not None and crop.size == (128, 128)
    assert crop_for_inference(_scene(tomato=False), "neutral") is None
    with pytest.raises(ValueError):
        crop_for_inference(_scene(), "blur")
