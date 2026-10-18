"""Memuat config.yaml. Semua parameter proyek dibaca dari sini, bukan ditulis di kode."""
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path=None) -> dict:
    with open(path or DEFAULT_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
