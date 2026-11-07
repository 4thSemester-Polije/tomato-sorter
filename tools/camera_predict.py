"""Uji webcam -> vision -> API model TANPA Arduino, dengan hasil ikut tampil di dashboard.

Menjalankan jalur yang sama dengan controller (foto segar, segmentasi, unggah JPEG ke ml/api.py), lalu
mengirim hasilnya ke dashboard lewat jalur yang sama dengan controller (/api/ingest/*): tomat masuk
database, kamera tampil di panel kamera, dan kejadian tercatat di log. Tomat dari uji kamera ditandai
dengan firmware_id kosong (tomat dari mesin selalu punya firmware_id).

    1) python -m uvicorn ml.api:app --port 9000      (API model)
    2) python -m dashboard                            (opsional; tanpa ini hasil hanya di terminal)
    3) python -m tools.camera_predict                 (--no-dashboard untuk tidak mengirim)
Tombol: SPASI = foto dan klasifikasikan | q = keluar
Jangan jalankan bersamaan dengan controller: webcam hanya bisa dibuka satu program.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
TAG = "[Uji kamera]"


def analyze(frame, background, mm_per_pixel, predict: Callable[[bytes], dict]) -> dict[str, Any]:
    """Segmentasi lalu klasifikasi satu frame. Bila tidak ada tomat valid, API model tidak dipanggil."""
    from controller.imaging import encode_jpeg
    from vision.features import extract_features
    from vision.segmentation import segment

    mask = segment(frame, background)
    try:
        features = extract_features(frame, mask, mm_per_pixel)
    except ValueError:
        return {"valid": False, "mask": mask, "features": None, "result": None, "latency_ms": None}
    started = time.perf_counter()
    result = predict(encode_jpeg(frame))
    return {"valid": True, "mask": mask, "features": features, "result": result,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1)}


def format_summary(outcome: dict[str, Any]) -> str:
    if not outcome["valid"]:
        return "Tidak ada tomat valid di gambar (controller tidak akan membalas; firmware akan GRADE_TIMEOUT)."
    r = outcome["result"]
    return (f"GRADE {r['grade']} | kematangan: {r['ripeness']['class']} ({r['ripeness']['confidence']:.0%}) | "
            f"cacat: {r['defect']['class']} ({r['defect']['confidence']:.0%}) | "
            f"model {r.get('architecture')} | waktu API {outcome['latency_ms']} ms")


def build_tomato_payload(outcome: dict[str, Any], image_path: str | None) -> dict[str, Any]:
    """Isi POST /api/ingest/tomatoes untuk hasil uji kamera (tanpa firmware_id)."""
    r = outcome["result"]
    payload = {k: r.get(k) for k in ("ripeness", "defect", "architecture")}
    payload.update(grade=r["grade"], latency_ms=outcome["latency_ms"], image_path=image_path)
    return payload


def _relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> None:
    import cv2

    from controller.camera import Camera
    from controller.config import load_config, load_env
    from controller.dashboard_client import DashboardClient, check_dashboard
    from controller.imaging import encode_jpeg, write_jpeg_atomic
    from controller.ml_client import check_api, make_image_predictor
    from vision.debug import draw_debug

    ap = argparse.ArgumentParser(prog="python -m tools.camera_predict", description=__doc__.splitlines()[0])
    ap.add_argument("--no-dashboard", action="store_true", help="jangan kirim hasil ke dashboard")
    args = ap.parse_args()

    load_env()
    cfg = load_config()
    ml = cfg.get("ml") or {}
    if not ml.get("api_url"):
        raise SystemExit("ml.api_url di config.yaml masih null. Isi dengan alamat API model, mis. http://127.0.0.1:9000")
    ok, message = check_api(ml["api_url"], ml.get("architecture"))
    if not ok:
        raise SystemExit(message)
    print(message)
    predict = make_image_predictor(ml["api_url"], ml.get("architecture"), ml.get("timeout_s", 10))

    client = None
    if not args.no_dashboard:
        ok, message = check_dashboard(cfg["dashboard"]["url"])
        print(message if ok else f"PERINGATAN: {message}\n  Hasil uji hanya tampil di terminal.")
        if ok:
            client = DashboardClient.from_config(cfg).start()
            client.event("INFO", "controller", f"{TAG} dimulai (tanpa Arduino)")

    bg_path = Path(cfg["paths"]["background_image"])
    bg_path = bg_path if bg_path.is_absolute() else ROOT / bg_path
    background = cv2.imread(str(bg_path)) if bg_path.is_file() else None
    print("Latar kosong:", bg_path if background is not None else "tidak ada (memakai segmentasi saturasi)")

    cam = cfg["camera"]
    camera = Camera(cam["index"], cam["width"], cam["height"], cam["flush_frames"])
    debug_dir = Path(cfg["paths"]["debug_dir"])
    debug_dir = debug_dir if debug_dir.is_absolute() else ROOT / debug_dir
    frame_interval = 1.0 / max(1, cfg["dashboard"]["stream_fps"])
    last_frame_sent = last_beat = 0.0
    last_text = "SPASI = klasifikasi | q = keluar"
    try:
        while True:
            view = camera.read()
            now = time.monotonic()
            if client is not None:
                if now - last_frame_sent >= frame_interval:
                    jpeg = encode_jpeg(view)
                    if jpeg:
                        client.frame(jpeg)
                    last_frame_sent = now
                if now - last_beat >= 1.0:  # heartbeat agar status dashboard tidak dianggap usang
                    client.status(serial_ok=False, camera_ok=True)
                    last_beat = now
            cv2.putText(view, last_text[:70], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("camera_predict", view)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key != ord(" "):
                continue
            frame = camera.read_fresh()
            try:
                outcome = analyze(frame, background, cfg["calibration"]["mm_per_pixel"], predict)
            except (OSError, ValueError) as exc:
                last_text = f"API gagal: {exc.__class__.__name__}"
                print(last_text, exc)
                if client is not None:
                    client.event("ERROR", "controller", f"{TAG} API model gagal ({exc.__class__.__name__})")
                continue
            last_text = format_summary(outcome)
            print(last_text)
            image_path = None
            if outcome["valid"]:
                debug = draw_debug(frame, outcome["mask"], outcome["features"])
                path = debug_dir / f"UJI_{int(time.time() * 1000)}.jpg"
                write_jpeg_atomic(path, debug)
                image_path = _relative(path)
                print("  foto debug:", image_path)
            if client is not None:
                if outcome["valid"]:
                    client.tomato(build_tomato_payload(outcome, image_path))
                    client.event("INFO", "controller", f"{TAG} {last_text}"[:500])
                else:
                    client.event("WARN", "controller", f"{TAG} tidak ada tomat valid di gambar")
    finally:
        if client is not None:
            client.status(camera_ok=False)
            client.event("INFO", "controller", f"{TAG} selesai")
            client.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
