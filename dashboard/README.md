# dashboard/

Antarmuka web untuk melihat status alat, kamera real-time, hasil grade, dan riwayat.
Stack: FastAPI + stream MJPEG + SQLite. Tanpa CDN dan tanpa internet (HTML, CSS, JS murni), jadi aman dipakai di PC alat.

## Menjalankan

```
pip install -r requirements.txt
python -m dashboard              # buka http://127.0.0.1:8000 (jalankan python -m controller.main di terminal lain)
```

Semua pengaturan (port, lokasi database, FPS stream, jendela grafik) ada di `config.yaml` bagian `dashboard`.

## Cara kerja

Dashboard TIDAK membuka serial atau webcam. Controller dan dashboard adalah dua proses yang bertukar data lewat
dua hal saja, jadi tim lain tidak perlu menunggu dashboard dan sebaliknya:

| Apa | Siapa menulis | Isi |
|---|---|---|
| `dashboard.database` (SQLite) | controller | status alat dan satu baris per tomat |
| `dashboard.live_frame` (JPEG) | controller | frame pratinjau terbaru, ditimpa terus |

Tampilan yang dihasilkan: pita status yang mengikuti protokol (READY, DETECT, GRADING, ACK, DONE, atau ERR),
kamera, tomat terakhir (foto segmentasi dan fitur), proporsi grade A/B/C, laju per menit, daftar error, dan riwayat
dengan filter serta ekspor CSV.

## Menyambungkan controller

`controller/main.py` sudah menulis ke database dan ke `dashboard.live_frame`. Jalankan dua program di dua terminal:

```
python -m controller.main      # terminal 1: serial + kamera + vision + ml
python -m dashboard            # terminal 2: dashboard, buka http://127.0.0.1:8000
```

Syarat sebelum controller bisa jalan:

- `capture.settle_after_detect_s` di `config.yaml` harus diisi (sekarang `null`). Untuk uji awal boleh `1.0`.
- `serial.port` di `config.yaml` sesuai port ESP32 (mis. `COM3`).
- `ml/predict.py` harus sudah diimplementasikan oleh tim sistem cerdas. Selama masih `NotImplementedError`, controller tidak mengirim GRADE dan firmware berakhir di `ERR GRADE_TIMEOUT`.
- Opsional: foto latar kosong di `data/raw/background.jpg` (tombol `b` di `tools/capture_dataset`). Tanpa itu dipakai segmentasi saturasi.

Yang dicatat controller:

| Pesan dari Arduino | Status di dashboard | Database |
|---|---|---|
| `READY` | Siap | - |
| `DETECT,<id>` | Tomat terdeteksi, lalu Menghitung grade | - |
| `ACK,<id>` | Mengantar | - |
| `DONE,<id>,<g>` | Selesai | satu baris hasil (fitur, grade, foto) |
| `ERR,<id>,<kode>` | Error + kodenya | satu baris error |

Tomat tanpa objek valid di gambar (mis. hanya tangan) sengaja tidak dibalas, sesuai `docs/ALUR_ALAT.md`.
Heartbeat status dikirim tiap iterasi loop (sekitar sekali per detik).

## Catatan desain

- `DONE` ditampilkan sebagai "Selesai" dengan keterangan menurut waktu tempuh, bukan "terkonfirmasi" (lihat catatan di `docs/PROTOCOL.md`).
- Grade selalu tampil sebagai huruf, tidak hanya warna. Merah tomat sengaja tidak dipakai untuk menandai grade jelek, karena grade C juga mencakup tomat mentah.
- Nama kolom fitur di database dijaga sama dengan `vision.features.FEATURE_NAMES` lewat `tests/test_dashboard.py`.
- Foto hanya dilayani dari folder `data/`.
- Belum ada login. Jalankan di `127.0.0.1` (bawaan). Bila host diubah ke `0.0.0.0` agar bisa dibuka dari perangkat lain, siapa pun di jaringan yang sama bisa melihat dashboard.

## Belum dikerjakan

- Tombol Start/Stop/Reset dari dashboard (perlu jalur perintah dari dashboard ke controller).
- Tab kalibrasi (foto latar, `mm_per_pixel`, ambang warna).