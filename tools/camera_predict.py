"""Uji webcam -> vision -> API model TANPA Arduino dan tanpa Postman.

Menjalankan jalur yang sama dengan controller (foto segar, segmentasi, unggah JPEG ke ml/api.py), jadi
hasilnya mewakili apa yang akan dilihat controller. Jalankan API model dulu:
    python -m uvicorn ml.api:app --port 9000
lalu:
    python -m tools.camera_predict
Tombol:  SPASI = foto dan klasifikasikan | q = keluar
Jangan jalankan bersamaan dengan controller: webcam hanya bisa dibuka satu program.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent


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


def main() -> None:
    import cv2

    from controller.camera import Camera
    from controller.config import load_config
    from controller.imaging import write_jpeg_atomic
    from controller.ml_client import check_api, make_image_predictor
    from vision.debug import draw_debug

    cfg = load_config()
    ml = cfg.get("ml") or {}
    if not ml.get("api_url"):
        raise SystemExit("ml.api_url di config.yaml masih null. Isi dengan alamat API model, mis. http://127.0.0.1:9000")
    ok, message = check_api(ml["api_url"], ml.get("architecture"))
    if not ok:
        raise SystemExit(message)
    print(message)
    predict = make_image_predictor(ml["api_url"], ml.get("architecture"), ml.get("timeout_s", 10))

    bg_path = Path(cfg["paths"]["background_image"])
    bg_path = bg_path if bg_path.is_absolute() else ROOT / bg_path
    background = cv2.imread(str(bg_path)) if bg_path.is_file() else None
    print("Latar kosong:", bg_path if background is not None else "tidak ada (memakai segmentasi saturasi)")

    cam = cfg["camera"]
    camera = Camera(cam["index"], cam["width"], cam["height"], cam["flush_frames"])
    debug_dir = Path(cfg["paths"]["debug_dir"])
    debug_dir = debug_dir if debug_dir.is_absolute() else ROOT / debug_dir
    last_text = "SPASI = klasifikasi | q = keluar"
    try:
        while True:
            view = camera.read()
            cv2.putText(view, last_text[:70], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("camera_predict", view)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                frame = camera.read_fresh()
                try:
                    outcome = analyze(frame, background, cfg["calibration"]["mm_per_pixel"], predict)
                except (OSError, ValueError) as exc:
                    last_text = f"API gagal: {exc.__class__.__name__}"
                    print(last_text, exc)
                    continue
                last_text = format_summary(outcome)
                print(last_text)
                debug = draw_debug(frame, outcome["mask"], outcome["features"]) if outcome["valid"] else frame
                path = debug_dir / f"camera_predict_{int(time.time() * 1000)}.jpg"
                write_jpeg_atomic(path, debug)
                print("  foto debug:", path)
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
