# Protokol Serial PC <-> Arduino v0.2 (DRAFT)

Status: draft. Akan direvisi setelah kode Arduino yang ada ditinjau dan posisi sensor/servo dikonfirmasi tim hardware.

## Aturan umum

- USB serial, 115200 baud, 8N1.
- Teks ASCII, satu pesan per baris, diakhiri `\n`. Field dipisah koma.
- ID tomat dibuat oleh PC pada setiap siklus (bilangan bulat >= 0).
- Grade hanya huruf `A`, `B`, atau `C`.
- Satu tomat per siklus (interlock): PC hanya mengirim `GRADE` setelah menerima `READY`.

## Pesan

| Arah | Pesan | Arti |
|---|---|---|
| PC -> Arduino | `GRADE,<id>,<A\|B\|C>` | Hasil klasifikasi tomat |
| Arduino -> PC | `ACK,<id>` | Perintah diterima |
| Arduino -> PC | `DONE,<id>,<grade>` | Tomat masuk wadah, conveyor berhenti |
| Arduino -> PC | `ERR,<id>,<kode>` | Kegagalan. Kode: `NO_ARRIVAL`, `JAM`, `BUSY`, `BAD_CMD` |
| Arduino -> PC | `READY` | Siap menerima tomat baru (dikirim saat boot dan setelah `DONE`/`ERR` selesai ditangani) |

Contoh siklus normal: `READY` -> `GRADE,12,B` -> `ACK,12` -> `DONE,12,B` -> `READY`.

## Kode error

| Kode | Penyebab |
|---|---|
| `NO_ARRIVAL` | Tomat tidak terdeteksi tiba dalam batas waktu |
| `JAM` | Tomat tidak masuk lubang/wadah dalam batas waktu |
| `BUSY` | `GRADE` diterima saat Arduino belum `READY` |
| `BAD_CMD` | Pesan tidak dikenal atau format salah |

Setiap `ERR` yang berasal dari fase gerak membuat Arduino menghentikan conveyor.

## Mode debug manual

Di serial monitor, mengetik `A`, `B`, atau `C` (tanpa ID) tetap diterima untuk menguji servo. Perintah lama (`matang`/`setengah`/`busuk`) dihapus agar konsisten dengan rubrik.

## Batas waktu

Nilai batas waktu tiap tahap diukur dari alat nyata dan disimpan di `config.yaml` bagian `timeouts_s`. Jangan menulis angka waktu langsung di kode.
