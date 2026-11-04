from io import BytesIO

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
