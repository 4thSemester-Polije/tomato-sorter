"""Program utama di PC (TODO: integrator).

Siklus per tomat (protokol v0.3):
  1. Tunggu DETECT,<id> dari Arduino (IR terpicu, konveyor berhenti).
  2. Tunggu capture.settle_after_detect_s agar tomat diam.
  3. camera.read_fresh() -> gambar segar.
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
