# Tomato Sorter

Rancang Bangun Sistem Pemilahan Tomat Otomatis Berdasarkan Tingkat Kematangan dan Cacat Permukaan Menggunakan Pengolahan Citra dan Sistem Cerdas Berbasis Mikrokontroler

Proyek terintegrasi untuk tiga mata kuliah: Workshop Sistem Tertanam, Pengolahan Citra / Computer Vision, dan Sistem Cerdas.

## Alur kerja alat (ringkas)

1. Sensor IR mendeteksi tomat, konveyor berhenti, firmware mengirim `DETECT,<id>`.
2. Webcam mengambil gambar, modul `vision` mengekstrak fitur.
3. Model (`ml`) menentukan grade A/B/C berdasarkan rubrik (`docs/RUBRIK.md`).
4. `controller` (PC) mengirim `GRADE,<id>,<A|B|C>` ke Arduino/ESP32 lewat USB serial (`docs/PROTOCOL.md`).
5. Firmware: pusher mendorong tomat, konveyor berjalan, gate sesuai grade dibuka, lalu melapor `DONE`. Alur rinci: `docs/ALUR_ALAT.md`.

## Struktur repositori

| Folder / file | Fungsi |
|---|---|
| `firmware/` | Kode Arduino/ESP32: sensor, motor conveyor, servo, state machine, protokol serial |
| `vision/` | Pengolahan citra: segmentasi tomat dan ekstraksi fitur |
| `ml/` | Sistem cerdas: aturan rubrik, training ResNeXt-101, evaluasi, API prediksi |
| `controller/` | Program utama di PC: webcam, deteksi kehadiran tomat, vision, model, komunikasi serial |
| `dashboard/` | Antarmuka web (prioritas terakhir) |
| `data/` | `labels.csv` (di Git). Foto mentah di `data/raw/` TIDAK di-commit |
| `models/` | Model hasil pelatihan (tidak di-commit secara default) |
| `docs/` | Rubrik, protokol serial, alur alat, panduan kontribusi, [panduan ML/API](docs/ML_RESNEXT101.md) |
| `tests/` | Pengujian otomatis (pytest) |
| `tools/` | Skrip bantu: `capture_dataset` (foto dataset), `extract_features_batch` (foto ke fitur), `send_grade` (uji protokol) |
| `config.yaml` | Semua parameter (port, kamera, ambang, waktu). Jangan menulis angka di kode |

## Mulai cepat (PC)

```
python -m venv .venv
# Windows:   .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

Ambil foto dan ekstrak fitur:

```
python -m tools.capture_dataset --start-id 1
python -m tools.extract_features_batch
```

Uji protokol ke firmware (tanpa vision dan model):

```
python -m tools.send_grade A --port COM3
```

## Aturan tim

Lihat `docs/CONTRIBUTING.md`. Ringkasnya: tidak push langsung ke `main`, satu branch per tugas, Pull Request di-review minimal satu orang.
