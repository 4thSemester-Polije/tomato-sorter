"""Klien HTTP controller -> ml/api.py (POST /predict, unggah JPEG). Hanya pustaka standar."""
from __future__ import annotations

import json
import uuid
from urllib import request
from urllib.parse import urlencode


def build_multipart(
    jpeg: bytes, filename: str = "frame.jpg", frame_jpeg: bytes | None = None
) -> tuple[bytes, str]:
    """Badan multipart: `file` (gambar untuk model kematangan) dan, opsional, `frame` (frame mentah untuk model cacat)."""
    boundary = uuid.uuid4().hex

    def part(name: str, upload_name: str, data: bytes) -> bytes:
        head = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{upload_name}\"\r\n"
            "Content-Type: image/jpeg\r\n\r\n"
        ).encode()
        return head + data + b"\r\n"

    body = part("file", filename, jpeg)
    if frame_jpeg is not None:
        body += part("frame", "raw.jpg", frame_jpeg)
    return body + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def make_image_predictor(api_url: str, architecture: str | None = None, timeout_s: float = 10.0):
    """Hasilkan fungsi f(jpeg_bytes) -> dict respons /predict. Melempar OSError/ValueError bila gagal."""
    url = api_url.rstrip("/") + "/predict" + ("?" + urlencode({"architecture": architecture}) if architecture else "")

    def predict(jpeg: bytes, frame_jpeg: bytes | None = None) -> dict:
        body, ctype = build_multipart(jpeg, frame_jpeg=frame_jpeg)
        req = request.Request(url, data=body, headers={"Content-Type": ctype}, method="POST")
        with request.urlopen(req, timeout=timeout_s) as resp:  # URLError/HTTPError adalah subclass OSError
            result = json.loads(resp.read().decode("utf-8"))
        if result.get("grade") not in ("A", "B", "C"):
            raise ValueError(f"respons ML tidak valid: {str(result)[:120]}")
        return result

    predict.accepts_frame = True  # pemanggil boleh menyertakan frame_jpeg; predictor lain (mis. palsu di tes) tidak
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
    message = f"API model siap ({chosen})"
    head = body.get("defect_head")
    if head:  # tampilkan model cacat mana yang aktif, supaya salah konfigurasi langsung terlihat
        message += (
            f" | cacat: masukan {head.get('trained_input')}, ambang {head.get('threshold')}, "
            f"ikut grade: {'ya' if head.get('used_for_grade') else 'tidak'}"
        )
    return True, message
