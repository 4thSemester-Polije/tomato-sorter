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


def check_api(api_url: str, architecture: str | None = None, timeout_s: float = 5.0) -> tuple[bool, str]:
    """Periksa GET /health sebelum controller mulai, supaya kegagalan terlihat sejak awal, bukan pada tomat pertama."""
    url = api_url.rstrip("/") + "/health"
    try:
        with request.urlopen(url, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError) as e:
        return False, (f"API model tidak bisa dihubungi di {url} ({e.__class__.__name__}). "
                       "Jalankan dulu: python -m uvicorn ml.api:app --port 9000")
    chosen = architecture or body.get("architecture")
    info = (body.get("architectures") or {}).get(chosen) or {"status": body.get("status"),
                                                              "missing_files": body.get("missing_files", [])}
    if info.get("status") != "ready":
        missing = ", ".join(info.get("missing_files") or []) or "tidak diketahui"
        return False, f"Model {chosen} belum siap di API (file hilang: {missing}). Salin file model ke folder models/."
    return True, f"API model siap ({chosen})"
