"""Menulis frame pratinjau untuk stream dashboard.

Ditulis ke file sementara lalu diganti (os.replace) supaya pembaca tidak pernah melihat JPEG setengah jadi.
Dipakai oleh controller (kamera asli) dan oleh dashboard/dummy.py (kamera tiruan).
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2


def write_jpeg_atomic(path, image, quality: int = 80) -> bool:
    """Simpan gambar BGR sebagai JPEG secara atomik. False bila encode gagal."""
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(buf.tobytes())
    os.replace(tmp, path)
    return True