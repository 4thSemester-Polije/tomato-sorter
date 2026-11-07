"""Pemeriksaan sebelum menjalankan ketiganya (API model, dashboard, controller + hardware).

    python -m tools.check_stack

Jalankan SEBELUM controller (kamera dan port serial hanya bisa dibuka satu program). Keluar dengan kode 1
bila ada yang GAGAL; PERINGATAN tidak menggagalkan.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, NamedTuple

ROOT = Path(__file__).resolve().parent.parent
OK, WARN, FAIL = "OK", "PERINGATAN", "GAGAL"


class Check(NamedTuple):
    name: str
    status: str
    detail: str


def run_checks(cfg: dict, *, check_api: Callable, check_dashboard: Callable, serial_ports: Callable,
               camera_opens: Callable, env: dict | None = None, exists: Callable = lambda p: Path(p).is_file()) -> list[Check]:
    """Semua ketergantungan ke dunia luar disuntikkan agar bisa diuji tanpa hardware."""
    env = os.environ if env is None else env
    checks: list[Check] = []

    def add(name, ok, detail, bad=FAIL):
        checks.append(Check(name, OK if ok else bad, detail))

    from controller.main import validate_config
    try:
        validate_config(cfg)
        add("Config", True, "capture.settle_after_detect_s valid")
    except SystemExit as exc:
        add("Config", False, str(exc))

    ml = cfg.get("ml") or {}
    if ml.get("api_url"):
        ok, msg = check_api(ml["api_url"], ml.get("architecture"))
        add("API model", ok, msg)
    else:
        add("API model", False, "ml.api_url masih null: controller tidak memakai model ResNeXt")

    ok, msg = check_dashboard(cfg["dashboard"]["url"])
    add("Dashboard", ok, msg, bad=WARN)  # dashboard mati tidak menghentikan pemilahan

    bg = cfg["paths"]["background_image"]
    bg = bg if Path(bg).is_absolute() else ROOT / bg
    add("Foto latar kosong", exists(bg), str(bg) if exists(bg) else
        f"tidak ada ({bg}); segmentasi memakai cadangan saturasi. Buat dengan: python -m tools.capture_dataset, tombol b", bad=WARN)

    port = cfg["serial"]["port"]
    ports = serial_ports()
    detected = ", ".join(p for p, _ in ports) or "tidak ada"
    add("Port serial", any(p == port for p, _ in ports),
        f"{port} terdeteksi" if any(p == port for p, _ in ports) else
        f"{port} tidak terdeteksi (terdeteksi: {detected}). Isi serial.port di config.local.yaml")

    index = cfg["camera"]["index"]
    add("Kamera", camera_opens(index),
        f"indeks {index} terbuka" if camera_opens(index) else
        f"indeks {index} tidak bisa dibuka (dipakai program lain atau salah indeks). Jalankan: python -m tools.hardware_check")

    add("GEMINI_API_KEY (chatbot)", bool(env.get("GEMINI_API_KEY")),
        "terisi" if env.get("GEMINI_API_KEY") else "kosong: chatbot dashboard nonaktif (isi di .env)", bad=WARN)
    return checks


def format_checks(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines = [f"[{c.status:<10}] {c.name:<{width}}  {c.detail}" for c in checks]
    failed = sum(c.status == FAIL for c in checks)
    warned = sum(c.status == WARN for c in checks)
    lines.append("")
    lines.append("Siap dijalankan." if not failed else f"{failed} pemeriksaan GAGAL: perbaiki dulu sebelum menjalankan controller.")
    if warned:
        lines.append(f"{warned} peringatan (tidak menghentikan).")
    return "\n".join(lines)


def _camera_opens(index: int) -> bool:
    import cv2

    cap = cv2.VideoCapture(index)
    try:
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        return bool(ok and frame is not None)
    finally:
        cap.release()


def main() -> None:
    from controller.config import load_config, load_env
    from controller.dashboard_client import check_dashboard
    from controller.ml_client import check_api
    from tools.hardware_check import list_serial_ports

    load_env()
    checks = run_checks(load_config(), check_api=check_api, check_dashboard=check_dashboard,
                        serial_ports=list_serial_ports, camera_opens=_camera_opens)
    print(format_checks(checks))
    sys.exit(1 if any(c.status == FAIL for c in checks) else 0)


if __name__ == "__main__":
    main()
