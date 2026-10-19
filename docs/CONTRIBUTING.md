# Panduan Kontribusi (GitHub Flow)

1. `main` selalu dalam keadaan berjalan. Jangan push langsung ke `main`.
2. Satu tugas = satu branch, namanya `<modul>/<deskripsi-singkat>`, contoh:
   - `vision/segmentasi-hsv`
   - `firmware/state-machine`
   - `ml/baseline-random-forest`
3. Commit kecil dan sering, pesan commit singkat dan jelas (boleh Bahasa Indonesia).
4. Buka Pull Request ke `main`. Minimal satu anggota lain me-review sebelum merge.
5. Jalankan `python -m pytest` sebelum membuka Pull Request.
6. Jangan commit foto mentah (`data/raw/`), password, atau file besar. Foto dibagikan lewat folder bersama.
7. Jangan menulis angka kalibrasi di kode. Taruh di `config.yaml`.
8. Jangan mengubah `docs/PROTOCOL.md` atau nama fitur di `vision/features.py` tanpa memberi tahu seluruh tim, karena itu kontrak antarmuka antar modul.

## Pengaturan GitHub yang disarankan

- Settings -> Branches -> Add rule untuk `main`: wajib Pull Request dan minimal 1 approval.
- Tambahkan semua anggota sebagai collaborator.
- Gunakan tab Projects (kanban: To Do, Doing, Review, Done) untuk membagi tugas.

## Menerapkan file .patch

Dari root repo (PowerShell atau terminal):

```
git switch -c <modul>/<deskripsi>
git apply --check <file>.patch     # uji kering, tidak mengubah apa pun
git apply <file>.patch
python -m pytest
git add -A
git commit
```

- Jangan menerapkan patch langsung di `main`. Buat branch, lalu Pull Request.
- Terapkan patch berurutan sesuai nomor. Setiap patch menyebut basisnya.
- Bila `--check` gagal, jangan dipaksa: file kemungkinan sudah berubah dari basis patch. Kirim pesan error dan versi file terbaru.
- Bila error disebabkan line ending (Windows), coba `git apply --ignore-whitespace`. `.gitattributes` repo ini memaksa LF.
- Commit atas nama sendiri. Cantumkan penggunaan bantuan AI sesuai ketentuan kampus.
