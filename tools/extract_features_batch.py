"""Jalankan segmentasi + ekstraksi fitur untuk semua foto di data/photos.csv.

Keluaran: data/features.csv dan gambar debug di data/debug/.
Jalankan:  python -m tools.extract_features_batch
"""
import argparse
import csv
import os

from vision.features import FEATURE_NAMES, extract_features
from vision.segmentation import segment

FEATURE_FIELDS = ["id_tomat", "nama_file", "seg_ok", *FEATURE_NAMES]


def process_photos(rows, raw_dir, background=None, mm_per_pixel=None, debug_dir=None):
    """rows: daftar dict berisi id_tomat dan nama_file. Mengembalikan daftar dict sesuai FEATURE_FIELDS."""
    import cv2

    from vision.debug import draw_debug

    results = []
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
    for r in rows:
        out = {"id_tomat": r["id_tomat"], "nama_file": r["nama_file"], "seg_ok": 0}
        out.update({k: None for k in FEATURE_NAMES})
        img = cv2.imread(os.path.join(raw_dir, r["nama_file"]))
        if img is None:
            results.append(out)
            continue
        mask = segment(img, background)
        feats = None
        try:
            feats = extract_features(img, mask, mm_per_pixel)
            out.update(feats)
            out["seg_ok"] = 1
        except ValueError:
            pass
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, r["nama_file"]), draw_debug(img, mask, feats))
        results.append(out)
    return results


def main():
    import cv2

    from controller.config import load_config

    p = argparse.ArgumentParser(description="Ekstrak fitur dari semua foto di photos.csv")
    p.add_argument("--background", default=None, help="Foto latar kosong (default dari config.yaml)")
    args = p.parse_args()

    cfg = load_config()
    paths = cfg["paths"]
    bg_path = args.background or paths["background_image"]
    background = cv2.imread(bg_path) if os.path.exists(bg_path) else None
    if background is None:
        print(f"Peringatan: foto latar kosong tidak ditemukan ({bg_path}). Memakai segmentasi saturasi (cadangan).")

    with open(paths["photos_csv"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    results = process_photos(rows, paths["raw_images"], background,
                             cfg["calibration"]["mm_per_pixel"], paths["debug_dir"])

    with open(paths["features_csv"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FEATURE_FIELDS)
        w.writeheader()
        w.writerows(results)
    ok = sum(r["seg_ok"] for r in results)
    print(f"{ok}/{len(results)} foto tersegmentasi. Periksa gambar debug di {paths['debug_dir']}/")


if __name__ == "__main__":
    main()
