"""Memuat config.yaml. Semua parameter proyek dibaca dari sini, bukan ditulis di kode.

Nilai yang berbeda per laptop (port serial, indeks kamera) ditaruh di config.local.yaml di folder yang
sama. File itu tidak masuk Git dan hanya berisi bagian yang berbeda; isinya menimpa config.yaml.
"""
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
LOCAL_NAME = "config.local.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path=None) -> dict:
    path = Path(path) if path else DEFAULT_PATH
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    local = path.with_name(LOCAL_NAME)
    if local.is_file():
        with open(local, "r", encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        if not isinstance(override, dict):
            raise ValueError(f"{local.name} harus berisi pemetaan YAML (kunci: nilai)")
        _deep_merge(cfg, override)
    return cfg


def load_env() -> None:
    """Muat .env di root proyek (GEMINI_API_KEY, DASHBOARD_INGEST_KEY) ke environment proses ini.

    Variabel yang sudah ada di terminal menang atas isi .env. Dipakai controller dan alat uji agar
    memakai kunci yang sama dengan dashboard (yang juga membaca .env); tanpa ini kunci di .env hanya
    dikenali dashboard dan semua kiriman controller ditolak (HTTP 401).
    """
    from dotenv import load_dotenv

    load_dotenv(DEFAULT_PATH.parent / ".env")
