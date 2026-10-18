# data/

- `labels.csv`: label tiap tomat. DI-COMMIT ke Git.
- `raw/`: foto mentah. TIDAK di-commit (lihat `.gitignore`). Bagikan lewat folder bersama tim.

## Penamaan foto

`T001_<urutan>_<yyyymmdd>.jpg`, ID (`T001`) sama dengan kolom `id_tomat` di `labels.csv`.
Semua foto satu tomat memakai ID yang sama (penting agar pembagian train/test dilakukan PER TOMAT, bukan per foto).

## Kolom labels.csv

`id_tomat, tanggal, nama_file, tingkat_kematangan, tingkat_cacat, grade, pelabel_1, pelabel_2, catatan`
