"""Tulis kontrak OpenAPI untuk frontend:  python -m dashboard.export_openapi  ->  docs/openapi.json"""
import json
from pathlib import Path

from controller.config import load_config
from dashboard.app import ROOT, create_app

if __name__ == "__main__":
    import tempfile

    cfg = load_config()
    cfg["dashboard"]["database"] = str(Path(tempfile.mkdtemp()) / "x.db")
    out = ROOT / "docs" / "openapi.json"
    out.write_text(json.dumps(create_app(cfg).openapi(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Ditulis:", out)
