# Tomato Sorter

Rancang Bangun Sistem Pemilahan Tomat Otomatis Berdasarkan Tingkat Kematangan dan Cacat Permukaan Menggunakan Pengolahan Citra dan Sistem Cerdas Berbasis Mikrokontroler

Proyek terintegrasi untuk tiga mata kuliah: Workshop Sistem Tertanam, Pengolahan Citra / Computer Vision, dan Sistem Cerdas.

## Alur kerja alat (ringkas)

1. Tomat ditaruh manual di stasiun foto (terpisah dari conveyor).
2. Webcam mengambil gambar, modul `vision` mengekstrak fitur.
3. Model (`ml`) menentukan grade A/B/C berdasarkan rubrik (`docs/RUBRIK.md`).
4. `controller` (PC) mengirim `GRADE,<id>,<A|B|C>` ke Arduino lewat USB serial (`docs/PROTOCOL.md`).
5. Arduino: servo 1 menendang tomat ke conveyor, servo 2 mengarahkan ke wadah sesuai grade, sensor IR mengonfirmasi, lalu melapor `DONE`.

## Struktur repositori

| Folder / file | Fungsi |
|---|---|
| `firmware/` | Kode Arduino/ESP32: sensor, motor conveyor, servo, state machine, protokol serial |
| `vision/` | Pengolahan citra: segmentasi tomat dan ekstraksi fitur |
| `ml/` | Sistem cerdas: aturan rubrik, pelatihan, evaluasi, prediksi |
| `controller/` | Program utama di PC: webcam, deteksi kehadiran tomat, vision, model, komunikasi serial |
| `dashboard/` | Antarmuka web (prioritas terakhir) |
| `data/` | `labels.csv` (di Git). Foto mentah di `data/raw/` TIDAK di-commit |
| `models/` | Model hasil pelatihan (tidak di-commit secara default) |
| `docs/` | Rubrik, protokol serial, alur alat, panduan kontribusi, catatan uji |
| `tests/` | Pengujian otomatis (pytest) |
| `tools/` | Skrip bantu (mis. mengirim grade ke Arduino untuk uji protokol) |
| `config.yaml` | Semua parameter (port, kamera, ambang, waktu). Jangan menulis angka di kode |

## Mulai cepat (PC)

```
python -m venv .venv
# Windows:   .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

Uji protokol ke Arduino (tanpa vision dan model):

```
python -m tools.send_grade A --port COM3
```

## Aturan tim

Lihat `docs/CONTRIBUTING.md`. Ringkasnya: tidak push langsung ke `main`, satu branch per tugas, Pull Request di-review minimal satu orang.
