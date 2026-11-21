"""Ambil foto dari webcam, jalankan seluruh pipeline PCV (HSV + Tekstur CLAHE/Negatif/Blackhat-Tophat + LAB RGB),
dan simpan gambar hasil siap-uji untuk Postman (`data/debug/pcv/pcv_latest_enhanced.jpg`).

Jalankan:
    python -m tools.capture_pcv
Tombol:
    SPASI = ambil foto, proses PCV, & simpan hasil | q = keluar
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def process_and_save_pcv(frame_bgr, output_dir: str | Path, prefix: str | None = None) -> dict[str, Any]:
    """Jalankan seluruh fitur PCV pada 1 frame webcam dan simpan semua hasil visualnya ke output_dir."""
    import cv2

    from controller.imaging import write_jpeg_atomic
    from vision.features import extract_pcv_features

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = prefix or f"PCV_{int(time.time() * 1000)}"

    hasil = extract_pcv_features(frame_bgr, crop_size=300)

    paths = {
        "asli": out / f"{tag}_asli.jpg",
        "crop_1to1": out / f"{tag}_crop_1to1.jpg",
        "negatif_clahe": out / f"{tag}_negatif_clahe.jpg",
        "mask_cacat": out / f"{tag}_mask_cacat.jpg",
        "kontur_cacat": out / f"{tag}_kontur_cacat.jpg",
        "enhanced": out / f"{tag}_enhanced.jpg",
        "latest_enhanced": out / "pcv_latest_enhanced.jpg",
    }

    write_jpeg_atomic(paths["asli"], frame_bgr)
    write_jpeg_atomic(paths["crop_1to1"], hasil["crop_asli_bgr"])
    cv2.imwrite(str(paths["negatif_clahe"]), hasil["gray_clahe_negatif"])
    cv2.imwrite(str(paths["mask_cacat"]), hasil["cacat_mask"])
    write_jpeg_atomic(paths["kontur_cacat"], hasil["img_hasil_cacat_bgr"])
    write_jpeg_atomic(paths["enhanced"], hasil["img_enhanced_bgr"])
    write_jpeg_atomic(paths["latest_enhanced"], hasil["img_enhanced_bgr"])

    return {"hasil": hasil, "paths": paths}


def main() -> None:
    import cv2
    import numpy as np

    from controller.camera import Camera
    from controller.config import load_config

    parser = argparse.ArgumentParser(description="Tangkap foto webcam dan jalankan seluruh fitur PCV")
    parser.add_argument("--camera", type=int, default=None, help="Indeks kamera (default dari config.yaml)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Folder simpan hasil PCV (default data/debug/pcv)")
    args = parser.parse_args()

    cfg = load_config()
    cam_cfg = cfg["camera"]
    cam_idx = args.camera if args.camera is not None else cam_cfg["index"]
    out_dir = args.out_dir or (ROOT / cfg["paths"]["debug_dir"] / "pcv")

    camera = Camera(cam_idx, cam_cfg["width"], cam_cfg["height"], cam_cfg["flush_frames"])
    info_text = "SPASI = ambil & proses PCV | q = keluar"
    print(f"Kamera aktif (index {cam_idx}). Hasil PCV akan disimpan ke: {out_dir}")
    print("Tekan SPASI di jendela webcam untuk memotret & memproses PCV, atau tekan 'q' untuk keluar.")

    try:
        while True:
            view = camera.read()
            overlay = view.copy()
            cv2.putText(overlay, info_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Webcam PCV (SPASI = Foto, q = Keluar)", overlay)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key != ord(" "):
                continue

            frame = camera.read_fresh()
            saved = process_and_save_pcv(frame, out_dir)
            h = saved["hasil"]
            p = saved["paths"]

            print("\n=== HASIL PROSES PCV WEBCAM ===")
            print(
                f"Warna (HSV)   : Merah {h['persen_merah']:.2f}% | Oranye {h['persen_oranye']:.2f}% | "
                f"Kuning {h['persen_kuning']:.2f}% | Hijau {h['persen_hijau']:.2f}%"
            )
            print(
                f"Tekstur/Cacat : {h['persentase_cacat']:.2f}% "
                f"({h['jumlah_kontur_cacat']} kontur, luas tomat {h['luas_piksel']:.0f} px)"
            )
            print(f"File siap Postman : {p['latest_enhanced']}")
            print(f"File overlay cacat: {p['kontur_cacat']}")

            # Tampilkan pratinjau berdampingan: Enhanced RGB (ke AI) vs Overlay Kontur Cacat
            preview = np.hstack([h["img_enhanced_bgr"], h["img_hasil_cacat_bgr"]])
            cv2.imshow("Hasil PCV: Kiri = Enhanced LAB (ke AI) | Kanan = Kontur Cacat", preview)
            info_text = f"Tersimpan! Cacat {h['persentase_cacat']:.1f}% | Merah {h['persen_merah']:.0f}%"
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

