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
        lambda image, models_dir: {
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
