"""Program utama di PC (TODO: integrator).

Siklus per tomat:
  1. Tunggu Arduino melapor READY.
  2. Tunggu tomat hadir dan stabil di stasiun foto (presence).
  3. Ambil gambar segar dari webcam (camera.read_fresh).
  4. vision.segment -> vision.extract_features.
  5. ml.predict -> grade.
  6. Kirim GRADE,<id>,<grade>; tunggu ACK, lalu DONE atau ERR.
  7. Catat hasil (id, waktu, fitur, grade, latensi) untuk laporan.
"""
from controller.config import load_config


def main():
    cfg = load_config()
    print("Controller belum diimplementasikan. Konfigurasi terbaca:", list(cfg.keys()))


if __name__ == "__main__":
    main()
