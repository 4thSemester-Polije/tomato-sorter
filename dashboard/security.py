"""Otorisasi endpoint ingest (/api/ingest/*), tempat model, controller, dan firmware mengirim data."""
from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, Request

LOOPBACK = ("127.0.0.1", "::1", "localhost")


def require_ingest(request: Request) -> None:
    """Bila DASHBOARD_INGEST_KEY diatur: wajib header X-API-Key yang cocok.
    Bila tidak: hanya diterima dari mesin yang sama (loopback). Dengan begitu dashboard yang dibuka ke
    jaringan (host 0.0.0.0) tetap tidak bisa disuntik data oleh perangkat lain."""
    key = os.environ.get("DASHBOARD_INGEST_KEY")
    if key:
        if not hmac.compare_digest(request.headers.get("x-api-key", ""), key):
            raise HTTPException(status_code=401, detail="X-API-Key salah atau tidak ada")
        return
    host = request.client.host if request.client else ""
    if host not in LOOPBACK:
        raise HTTPException(status_code=403, detail="Ingest dari jaringan butuh DASHBOARD_INGEST_KEY di server")
