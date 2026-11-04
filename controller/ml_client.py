"""Klien HTTP controller -> ml/api.py (POST /predict, unggah JPEG). Hanya pustaka standar."""
from __future__ import annotations

import json
import uuid
from urllib import request
from urllib.parse import urlencode


def build_multipart(jpeg: bytes, filename: str = "frame.jpg") -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    head = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode()
    return head + jpeg + f"\r\n--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def make_image_predictor(api_url: str, architecture: str | None = None, timeout_s: float = 10.0):
    """Hasilkan fungsi f(jpeg_bytes) -> dict respons /predict. Melempar OSError/ValueError bila gagal."""
    url = api_url.rstrip("/") + "/predict" + ("?" + urlencode({"architecture": architecture}) if architecture else "")

    def predict(jpeg: bytes) -> dict:
        body, ctype = build_multipart(jpeg)
        req = request.Request(url, data=body, headers={"Content-Type": ctype}, method="POST")
        with request.urlopen(req, timeout=timeout_s) as resp:  # URLError/HTTPError adalah subclass OSError
            result = json.loads(resp.read().decode("utf-8"))
        if result.get("grade") not in ("A", "B", "C"):
            raise ValueError(f"respons ML tidak valid: {str(result)[:120]}")
        return result

    return predict
