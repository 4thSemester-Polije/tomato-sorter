# Runbook integrasi: kamera, hardware, model, dashboard, dan database

## Gambaran

```
 tomat -> IR -> ESP32 --DETECT,id--> controller --foto JPEG--> API model (:9000) -> grade
                  ^                      |  ^
                  +--GRADE,id,X----------+  +-- ACK / DONE / ERR dari ESP32
                                         |
                                         +--(HTTP /api/ingest/*)--> dashboard (:8000) --> data/tomato.db
                                         |   hasil tomat, pesan firmware, log, heartbeat, frame kamera
                                         +--> browser: status alat, kamera langsung, riwayat, log, chatbot
```

| Proses | Perintah | Memegang | Catatan |
|---|---|---|---|
| API model | `python -m uvicorn ml.api:app --port 9000` | file model di `models/` | Tunggu warm-up selesai |
| Dashboard | `python -m dashboard` | **database `data/tomato.db`** | Satu-satunya proses yang menulis database |
| Controller | `python -m controller.main` | webcam + port serial | Mengirim semuanya ke dashboard lewat HTTP |
| Uji kamera | `python -m tools.camera_predict` | webcam | Pengganti controller tanpa Arduino. **Jangan jalan bersamaan dengan controller** |

Controller dan alat uji tidak menyentuh database langsung; mereka mengirim ke dashboard. Karena itu database selalu konsisten dan tidak ada konflik penguncian SQLite.

## Persiapan satu kali

| Item | Cara |
|---|---|
| Firmware protokol v0.3 | Upload `firmware/tomato_sorter/tomato_sorter.ino`. Sketch lama (perintah `1/2/3`, log `[IR_DETECTED]`) tidak kompatibel |
| File model di `models/` | 4 file `*_resnext101.*` dari Drive tim (tidak masuk Git, lebih dari 100 MB) |
| Foto latar kosong | `python -m tools.capture_dataset`, stasiun kosong, tombol `b` -> `data/raw/background.jpg`. Ulangi bila kamera atau lampu bergeser |
| Port serial dan kamera laptop ini | `python -m tools.hardware_check`, isi `config.local.yaml` (tidak masuk Git): `serial: {port: COM5}` dan `camera: {index: 1}` |
| Kunci | File `.env` di root proyek (tidak masuk Git): `GEMINI_API_KEY=...` untuk chatbot. `DASHBOARD_INGEST_KEY=...` hanya perlu bila dashboard di PC lain; controller dan alat uji ikut membaca `.env` |

Pemetaan jalur: **A** = servo 3, **B** = servo 2, **C** = lurus (sama dengan `docs/PROTOCOL.md`).

## Tiga mode uji (dari paling sederhana)

| Mode | Yang jalan | Tujuan |
|---|---|---|
| 1. Hardware saja | `python -m tools.send_grade A` (tanpa kamera, model, dashboard) | Serial, servo, dan jalur fisik benar |
| 2. Kamera + model (+ dashboard) | API model, dashboard (opsional), `python -m tools.camera_predict` | Webcam -> vision -> model berfungsi; hasil ikut tampil di dashboard |
| 3. Lengkap | API model, dashboard, controller | Seluruh alur mesin |

Mode 2 tanpa dashboard: tambahkan `--no-dashboard`, hasil hanya di terminal. Tomat dari uji kamera tersimpan di database **tanpa `firmware_id`** (tomat dari mesin selalu punya) dan ikut terhitung di total.

## Prosedur end-to-end lengkap (mode 3)

Buka tiga terminal di root repo, venv aktif.

**Terminal 1, API model**
```powershell
python -m uvicorn ml.api:app --port 9000
```
Tunggu pesan warm-up selesai. Cek `http://127.0.0.1:9000/health` -> `status: ready`.

**Terminal 2, dashboard**
```powershell
python -m dashboard
```
Buka `http://127.0.0.1:8000`. Panel kamera masih kosong dan status alat belum terhubung; itu normal.

**Terminal 3, pemeriksaan lalu controller** (tutup Serial Monitor Arduino IDE dan `tools.camera_predict` dulu)
```powershell
python -m tools.check_stack
python -m controller.main
```
`check_stack` memeriksa config, API model, dashboard, foto latar, port serial, kamera, dan kunci chatbot. Semua `GAGAL` harus beres sebelum lanjut; `PERINGATAN` tidak menghentikan. Controller menunggu ESP32 (board reset saat port dibuka, tunggu `READY`), lalu mencetak `API model siap`, `Dashboard terhubung`, dan `Controller berjalan`.

**Satu tomat**: letakkan di stasiun foto. Urutan yang benar:

1. IR memicu; buzzer berbunyi; firmware mengirim `DETECT,<id>`; konveyor berhenti.
2. Controller menunggu `settle_after_detect_s`, mengambil foto segar, segmentasi, mengirim JPEG ke API model.
3. Controller mengirim `GRADE,<id>,<A|B|C>`; firmware membalas `ACK,<id>`; servo menggeser tomat ke jalur.
4. Setelah waktu tempuh habis firmware mengirim `DONE,<id>,<grade>` lalu `READY`.

**Yang harus terlihat di dashboard:**

| Elemen | Tampilan |
|---|---|
| Status alat | `READY` -> `DETECT` -> `GRADING` -> `ACK` -> `DONE` -> `READY`; serial dan kamera hijau |
| Kamera langsung | Gambar webcam bergerak |
| Riwayat | Baris baru: grade, kelas kematangan dan cacat beserta keyakinan, model `resnext101`, latensi, foto, status serial `DONE` |
| Log | "Tomat #n terdeteksi" dan "selesai diantar ke jalur X" |
| Total per grade dan grafik | Bertambah satu |
| Chatbot | "Berapa tomat per grade?" dijawab dari database |

## Memeriksa dan mereset database

Lihat 5 tomat terakhir (tanpa memasang sqlite3):

```powershell
python -c "import sqlite3; [print(r) for r in sqlite3.connect('data/tomato.db').execute('select id, firmware_id, grade, serial_status, architecture from tomatoes order by id desc limit 5')]"
```

Ekspor penuh: `http://127.0.0.1:8000/api/export.csv`.

Reset sebelum demo: hentikan dashboard, hapus `data/tomato.db` (opsional juga isi `data/debug/`), jalankan dashboard lagi; database dibuat ulang otomatis. Lakukan ini agar tomat uji tidak ikut terhitung.

## Anggaran waktu

Firmware menunggu `GRADE` paling lama `WAKTU_TUNGGU_GRADE` (15 s). Pemakaian per tomat: `settle_after_detect_s` + ambil frame + segmentasi + HTTP prediksi (sekitar 0,7 s sampai 1,3 s hanya model di CPU Colab; ukur di laptop ini). Bila total mendekati batas, naikkan `WAKTU_TUNGGU_GRADE` atau percepat salah satu langkah, lalu isi `timeouts_s.grade_wait` di `config.yaml`.

## Masalah umum

| Gejala | Penyebab | Tindakan |
|---|---|---|
| `could not open port COMx` | Port dipakai program lain atau salah | Tutup Serial Monitor; `python -m tools.hardware_check` |
| Kamera `GAGAL` di `check_stack` | Dipakai controller, `camera_predict`, atau aplikasi lain; salah indeks | Tutup yang lain; ganti `camera.index` di `config.local.yaml` |
| Controller jalan tapi dashboard kosong | Dashboard mati, atau kunci beda | Lihat pesan `[dashboard]` di terminal controller; samakan `DASHBOARD_INGEST_KEY` (HTTP 401) |
| `ERR,<id>,GRADE_TIMEOUT` | PC tidak membalas | Lihat terminal controller: API mati, tidak ada tomat valid di gambar, atau melebihi batas waktu |
| Controller berhenti dengan pesan "API model" | API belum jalan atau file model hilang | Jalankan terminal 1, tunggu warm-up |
| Hampir semua tomat menjadi C | Head defect di foto primer menandai tomat sehat sebagai reject | Masalah model, bukan integrasi; ditangani saat dataset diganti |
| `DONE` tetapi tomat tidak masuk wadah | `DONE` hanya berarti waktu tempuh habis | Kalibrasi `WAKTU_TEMPUH_*` di firmware |
