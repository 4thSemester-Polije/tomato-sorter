"""Penyimpanan dan encode JPEG untuk controller."""
from __future__ import annotations

import os
from pathlib import Path


def encode_jpeg(image, quality: int = 80) -> bytes | None:
    import cv2

    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def write_jpeg_atomic(path, image, quality: int = 80) -> bool:
    """Tulis ke file sementara lalu ganti (os.replace) agar pembaca tidak pernah melihat JPEG setengah jadi."""
    data = encode_jpeg(image, quality)
    if data is None:
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return True
