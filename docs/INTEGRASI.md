# Runbook integrasi model + hardware

Tujuan: satu tomat diletakkan di stasiun foto, IR memicu `DETECT`, webcam memotret, API model memberi grade, firmware menggerakkan servo ke jalur yang benar. Uji **bertahap**; jangan lanjut ke tahap berikut sebelum tahap sebelumnya lolos.

## Prasyarat

| Item | Cara |
|---|---|
| Firmware protokol v0.3 | Upload `firmware/tomato_sorter/tomato_sorter.ino` dari repo ini. Sketch lama (perintah `1/2/3`, `busuk/setengah/matang`, log `[IR_DETECTED]`) **tidak** kompatibel dengan controller |
| File model di `models/` | 4 file `*_resnext101.*` (dari Drive tim). Tidak masuk Git (lebih dari 100 MB) |
| Foto latar kosong | `python -m tools.capture_dataset`, kosongkan stasiun, tekan `b` (menghasilkan `data/raw/background.jpg`) |
| Port dan kamera laptop ini | `python -m tools.hardware_check`, lalu isi `config.local.yaml` (tidak masuk Git) |

Pemetaan jalur (sama dengan `docs/PROTOCOL.md`): **A** = servo 3 (kematangan dan mulus terbaik), **B** = servo 2, **C** = jalur lurus (mentah atau cacat berat). Jangan memakai pemetaan lama `1=A=busuk`; itu terbalik terhadap rubrik.

## Tahap 1: firmware dan serial (tanpa kamera, tanpa model)

1. Tutup Serial Monitor Arduino IDE (satu port hanya untuk satu program).
2. `python -m tools.send_grade A` lalu taruh tomat di depan IR. Terminal menampilkan `DETECT`, `ACK`, `DONE`.
3. Ulangi untuk `B` dan `C`. Periksa fisik: A masuk jalur servo 3, B servo 2, C lurus.

Lulus bila ketiga jalur benar dan tidak ada `ERR`.

## Tahap 2: API model

```powershell
python -m uvicorn ml.api:app --port 9000
```

Tunggu pesan warm-up selesai (server baru menerima permintaan setelah model termuat). `GET http://127.0.0.1:9000/health` harus `ready`. Uji `POST /predict` di Postman dengan satu foto tomat dan catat waktu respons.

## Tahap 3: kamera dan vision

`python -m tools.capture_dataset` untuk melihat gambar stasiun. Tomat harus terlihat utuh dan terang, kamera dan lampu tidak berubah posisi. Perubahan apa pun mewajibkan foto latar kosong baru.

## Tahap 4: end-to-end

Terminal 1: API model (tahap 2). Terminal 2: dashboard `python -m dashboard`. Terminal 3: `python -m controller.main`. Controller memeriksa API saat mulai dan berhenti dengan pesan jelas bila model belum siap.

Taruh satu tomat. Urutan yang benar: `DETECT` -> jeda `settle_after_detect_s` -> foto -> `GRADE` -> `ACK` -> `DONE` -> `READY`. Hasil dan foto muncul di dashboard.

## Anggaran waktu

Firmware menunggu `GRADE` paling lama `WAKTU_TUNGGU_GRADE` (15 s). Pemakaian waktu per tomat: `settle_after_detect_s` + ambil frame + segmentasi + HTTP prediksi (sekitar 0,7 s sampai 1,3 s hanya untuk model di CPU Colab; ukur di laptop ini). Bila total mendekati batas, naikkan `WAKTU_TUNGGU_GRADE` atau percepat salah satu langkah. Setelah terukur, isi `timeouts_s.grade_wait` di `config.yaml`.

## Masalah umum

| Gejala | Penyebab | Tindakan |
|---|---|---|
| `could not open port COMx` | Port dipakai program lain atau salah | Tutup Serial Monitor; jalankan `tools.hardware_check` |
| `ERR,<id>,GRADE_TIMEOUT` | PC tidak membalas | Lihat terminal controller: API mati, tomat tidak valid di gambar, atau waktu melebihi batas |
| Hampir semua tomat menjadi C | Head defect di foto primer menandai tomat sehat sebagai reject (recall healthy sekitar 49%) | Masalah model, bukan integrasi; ditangani saat dataset diganti |
| Controller berhenti dengan pesan "API model" | API belum jalan atau file model hilang | Jalankan tahap 2 |
| `DONE` muncul tapi tomat tidak masuk wadah | `DONE` hanya berarti waktu tempuh habis | Kalibrasi `WAKTU_TEMPUH_*` di firmware |
