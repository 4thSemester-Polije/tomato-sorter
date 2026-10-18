"""Pengambil foto dataset dari webcam, dengan ID tomat dan log di data/photos.csv.

Jalankan:  python -m tools.capture_dataset --start-id 1
Tombol:  SPASI = simpan foto tomat ini | n = tomat berikutnya | b = simpan foto LATAR KOSONG | q = keluar
Urutan: (1) kosongkan stasiun, tekan b. (2) taruh tomat, tekan SPASI beberapa kali (ubah posisi sedikit).
        (3) tekan n, ganti tomat, ulangi.
"""
import argparse
import csv
import os
from datetime import date

PHOTO_FIELDS = ["id_tomat", "nama_file", "tanggal"]


def tomato_id_str(n: int) -> str:
    return f"T{n:03d}"


def photo_filename(tomato_id: str, seq: int, day: str | None = None) -> str:
    day = day or date.today().strftime("%Y%m%d")
    return f"{tomato_id}_{seq:02d}_{day}.jpg"


def next_seq(raw_dir: str, tomato_id: str) -> int:
    """Nomor urut foto berikutnya untuk tomat ini (melanjutkan dari file yang sudah ada)."""
    best = 0
    if os.path.isdir(raw_dir):
        for name in os.listdir(raw_dir):
            parts = name.split("_")
            if len(parts) >= 3 and parts[0] == tomato_id:
                try:
                    best = max(best, int(parts[1]))
                except ValueError:
                    pass
    return best + 1


def append_photo_row(csv_path: str, row: dict) -> None:
    new_file = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PHOTO_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({k: row[k] for k in PHOTO_FIELDS})


def main():
    import cv2

    from controller.camera import Camera
    from controller.config import load_config

    p = argparse.ArgumentParser(description="Ambil foto dataset tomat dari webcam")
    p.add_argument("--start-id", type=int, default=1, help="Nomor tomat awal (T001 = 1)")
    p.add_argument("--camera", type=int, default=None, help="Indeks kamera (default dari config.yaml)")
    args = p.parse_args()

    cfg = load_config()
    raw_dir = cfg["paths"]["raw_images"]
    os.makedirs(raw_dir, exist_ok=True)
    cam_cfg = cfg["camera"]
    cam = Camera(args.camera if args.camera is not None else cam_cfg["index"],
                 cam_cfg["width"], cam_cfg["height"], cam_cfg["flush_frames"])

    n = args.start_id
    msg = ""
    try:
        while True:
            tid = tomato_id_str(n)
            frame = cam.read()
            view = frame.copy()
            cv2.putText(view, f"{tid}  foto berikutnya: {next_seq(raw_dir, tid)}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(view, "SPASI simpan | n berikut | b latar kosong | q keluar", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if msg:
                cv2.putText(view, msg, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("capture_dataset", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("n"):
                n += 1
                msg = ""
            elif key == ord("b"):
                path = cfg["paths"]["background_image"]
                cv2.imwrite(path, cam.read_fresh())
                msg = f"latar kosong disimpan: {path}"
            elif key == ord(" "):
                seq = next_seq(raw_dir, tid)
                name = photo_filename(tid, seq)
                cv2.imwrite(os.path.join(raw_dir, name), cam.read_fresh())
                append_photo_row(cfg["paths"]["photos_csv"],
                                 {"id_tomat": tid, "nama_file": name, "tanggal": date.today().isoformat()})
                msg = f"disimpan: {name}"
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
