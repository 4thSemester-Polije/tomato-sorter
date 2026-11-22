from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import ml.api as api


def _image_bytes() -> bytes:
    image = Image.new("RGB", (10, 10), color=(200, 30, 20))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_predict_endpoint_returns_grade_from_both_heads(monkeypatch):
    monkeypatch.setattr(
        api,
        "predict_image",
        lambda image, models_dir, architecture: {
            "model": "ResNeXt101",
            "input_size": [224, 224],
            "ripeness": {"class": "matang", "level": 1, "confidence": 0.9},
            "defect": {"class": "reject", "level": 3, "confidence": 0.8},
            "grade": "C",
        },
    )
    client = TestClient(api.app)

    response = client.post(
        "/predict",
        files={"file": ("tomato.jpg", _image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["grade"] == "C"
    assert response.json()["defect"]["level"] == 3


def test_predict_endpoint_rejects_non_image_upload():
    client = TestClient(api.app)

    response = client.post(
        "/predict",
        files={"file": ("not-image.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 422
    assert "not a readable" in response.json()["detail"]


def test_predict_endpoint_reports_missing_trained_models(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "DEFAULT_MODELS_DIR", tmp_path)
    client = TestClient(api.app)

    response = client.post(
        "/predict",
        files={"file": ("tomato.jpg", _image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 503
    assert "Trained model files are missing" in response.json()["detail"]


def test_predict_endpoint_passes_requested_architecture(monkeypatch):
    seen = {}

    def fake_predict(image, models_dir, architecture):
        seen["architecture"] = architecture
        return {"model": "ResNeXt50", "architecture": architecture, "grade": "A"}

    monkeypatch.setattr(api, "predict_image", fake_predict)
    client = TestClient(api.app)
    files = {"file": ("tomato.jpg", _image_bytes(), "image/jpeg")}

    response = client.post("/predict?architecture=resnext50", files=files)

    assert response.status_code == 200
    assert seen["architecture"] == "resnext50"


def test_predict_endpoint_uses_env_default_architecture(monkeypatch):
    seen = {}
    monkeypatch.setenv("TOMATO_ARCHITECTURE", "resnext50")
    monkeypatch.setattr(
        api,
        "predict_image",
        lambda image, models_dir, architecture: seen.setdefault("a", architecture) and {"grade": "B"},
    )
    client = TestClient(api.app)

    response = client.post(
        "/predict", files={"file": ("tomato.jpg", _image_bytes(), "image/jpeg")}
    )

    assert response.status_code == 200
    assert seen["a"] == "resnext50"


def test_predict_endpoint_rejects_unknown_architecture():
    client = TestClient(api.app)

    response = client.post(
        "/predict?architecture=resnet18",
        files={"file": ("tomato.jpg", _image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 422


def test_health_reports_each_architecture(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "DEFAULT_MODELS_DIR", tmp_path)
    for name in ("ripeness_resnext50.keras", "ripeness_resnext50.json",
                 "defect_resnext50.keras", "defect_resnext50.json"):
        (tmp_path / name).write_bytes(b"x")
    client = TestClient(api.app)

    body = client.get("/health").json()

    assert body["architectures"]["resnext50"]["status"] == "ready"
    assert body["architectures"]["resnext101"]["status"] == "model_missing"
    assert body["architecture"] == "resnext101"
    assert body["status"] == "model_missing"


def test_warm_up_returns_false_when_models_are_missing(tmp_path):
    assert api.warm_up(tmp_path, "resnext101") is False


def test_warm_up_runs_one_prediction_for_the_default_architecture(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setenv("TOMATO_ARCHITECTURE", "resnext50")
    monkeypatch.setattr(api, "predict_image", lambda image, models_dir, architecture: seen.update(a=architecture))

    assert api.warm_up(tmp_path) is True
    assert seen["a"] == "resnext50"


def test_server_startup_runs_warm_up_unless_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(api, "warm_up", lambda *a, **k: calls.append(1))

    with TestClient(api.app):
        pass
    assert calls == [1]

    monkeypatch.setenv("TOMATO_WARMUP", "0")
    with TestClient(api.app):
        pass
    assert calls == [1]


class _FakeModel:
    def __init__(self, probabilities):
        self._p = np.array([probabilities], dtype=np.float32)

    def predict(self, batch, verbose=0):
        return self._p


def _fake_models(monkeypatch, ripeness, defect):
    """ripeness: (matang, setengah_matang, mentah); defect: (healthy, reject)."""
    monkeypatch.setattr(
        api,
        "_load_models",
        lambda models_dir, architecture: (
            _FakeModel(ripeness),
            _FakeModel(defect),
            {"class_names": ["matang", "setengah_matang", "mentah"]},
            {"class_names": ["healthy", "reject"]},
            lambda pixels: pixels,
        ),
    )


def test_grade_ignores_defect_head_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("TOMATO_GRADE_USES_DEFECT", raising=False)
    _fake_models(monkeypatch, (0.9, 0.05, 0.05), (0.1, 0.9))

    result = api.predict_image(Image.new("RGB", (10, 10)), tmp_path, "resnext101")

    assert result["grade"] == "A"
    assert result["grade_basis"] == "ripeness"
    assert result["defect"]["class"] == "reject"  # keluaran CNN tetap utuh untuk alat evaluasi


def test_grade_uses_defect_head_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("TOMATO_GRADE_USES_DEFECT", "1")
    _fake_models(monkeypatch, (0.9, 0.05, 0.05), (0.1, 0.9))

    result = api.predict_image(Image.new("RGB", (10, 10)), tmp_path, "resnext101")

    assert result["grade"] == "C"
    assert result["grade_basis"] == "ripeness+defect"


def test_unripe_tomato_is_still_grade_c_without_defect_head(monkeypatch, tmp_path):
    monkeypatch.delenv("TOMATO_GRADE_USES_DEFECT", raising=False)
    _fake_models(monkeypatch, (0.05, 0.05, 0.9), (0.9, 0.1))

    assert api.predict_image(Image.new("RGB", (10, 10)), tmp_path, "resnext101")["grade"] == "C"


def test_predict_endpoint_hides_defect_guess_that_does_not_count(monkeypatch):
    monkeypatch.delenv("TOMATO_GRADE_USES_DEFECT", raising=False)
    _fake_models(monkeypatch, (0.1, 0.8, 0.1), (0.1, 0.9))
    client = TestClient(api.app)

    body = client.post("/predict", files={"file": ("t.jpg", _image_bytes(), "image/jpeg")}).json()

    assert body["grade"] == "B"
    assert body["defect"]["class"] == "tidak_dinilai" and body["defect"]["confidence"] == 0.0
    assert body["defect_raw"]["class"] == "reject"


def test_predict_endpoint_keeps_defect_when_it_counts(monkeypatch):
    monkeypatch.setenv("TOMATO_GRADE_USES_DEFECT", "1")
    _fake_models(monkeypatch, (0.9, 0.05, 0.05), (0.1, 0.9))
    client = TestClient(api.app)

    body = client.post("/predict", files={"file": ("t.jpg", _image_bytes(), "image/jpeg")}).json()

    assert body["grade"] == "C"
    assert body["defect"]["class"] == "reject" and "defect_raw" not in body


def test_predict_endpoint_with_apply_pcv_returns_pcv_summary(monkeypatch):
    seen = {}

    def fake_predict(image, models_dir, architecture):
        seen["size"] = image.size
        return {"model": "ResNeXt101", "architecture": architecture, "grade": "A"}

    monkeypatch.setattr(api, "predict_image", fake_predict)
    client = TestClient(api.app)

    response = client.post(
        "/predict?apply_pcv=true",
        files={"file": ("tomato.jpg", _image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert seen["size"] == (300, 300)
    assert body["grade"] == "A"
    assert "pcv" in body and "persen_merah" in body["pcv"] and "persentase_cacat" in body["pcv"]
