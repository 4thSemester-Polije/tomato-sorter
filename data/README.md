# data/

| File | Isi | Di Git? |
|---|---|---|
| `photos.csv` | Log foto (id_tomat, nama_file, tanggal), dibuat otomatis oleh `tools/capture_dataset.py` | Ya |
| `labels.csv` | Label PER TOMAT (satu baris per `id_tomat`), diisi manusia | Ya |
| `features.csv` | Fitur hasil `tools/extract_features_batch.py` (satu baris per foto) | Ya |
| `raw/` | Foto mentah dan `background.jpg` (foto stasiun kosong) | TIDAK (bagikan lewat folder bersama) |
| `debug/` | Gambar debug segmentasi | TIDAK |

## Penamaan foto

`T001_<urutan>_<yyyymmdd>.jpg`. ID yang sama untuk semua foto satu tomat, supaya pembagian train/test dilakukan PER TOMAT, bukan per foto.

## Kolom labels.csv

`id_tomat, pelabel_1, kematangan_1, cacat_1, pelabel_2, kematangan_2, cacat_2, tingkat_kematangan, tingkat_cacat, grade, catatan`

Kolom `_1` dan `_2` diisi masing-masing pelabel secara independen (tingkat 1-3 sesuai `docs/RUBRIK.md`). Kolom `tingkat_*` dan `grade` adalah keputusan akhir setelah perbedaan dibahas.

## Alur data

`capture_dataset` -> `raw/` + `photos.csv` -> `extract_features_batch` -> `features.csv` -> digabung dengan `labels.csv` lewat `id_tomat` -> pelatihan model.

