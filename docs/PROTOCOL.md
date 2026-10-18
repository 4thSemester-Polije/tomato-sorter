# Protokol Serial PC <-> Arduino/ESP32 v0.3 (DRAFT)

Status: draft, disusun dari kode firmware yang ada. Tim hardware menerapkannya di firmware.

## Aturan umum

- USB serial, 115200 baud, 8N1.
- Teks ASCII, satu pesan per baris, diakhiri `\n`. Field dipisah koma.
- ID tomat dibuat oleh firmware (penghitung) saat IR mendeteksi tomat, dan dikembalikan apa adanya oleh PC.
- Grade hanya huruf `A`, `B`, atau `C`.
- Baris yang diawali `#` adalah log debug dan diabaikan PC. Semua `Serial.print` debug di firmware WAJIB diawali `#`.

## Pesan

| Arah | Pesan | Arti |
|---|---|---|
| Arduino -> PC | `READY` | Saat boot dan setelah siklus selesai: mesin mencari tomat |
| Arduino -> PC | `DETECT,<id>` | IR terpicu, konveyor berhenti, menunggu `GRADE` |
| PC -> Arduino | `GRADE,<id>,<A\|B\|C>` | Hasil klasifikasi. `id` harus sama dengan `DETECT` |
| Arduino -> PC | `ACK,<id>` | `GRADE` diterima |
| Arduino -> PC | `DONE,<id>,<grade>` | Pengantaran selesai (lihat catatan) |
| Arduino -> PC | `ERR,<id>,<kode>` | Kegagalan |

Contoh siklus: `READY` -> `DETECT,12` -> `GRADE,12,B` -> `ACK,12` -> `DONE,12,B` -> `READY`.

Catatan `DONE`: dengan pengantaran berbasis waktu (tanpa sensor konfirmasi), `DONE` berarti "waktu tempuh habis dan gate ditutup", BUKAN "tomat terbukti masuk wadah". Jangan menyebutnya "terkonfirmasi" di laporan kecuali ada sensor di wadah.

## Kode error

| Kode | Penyebab |
|---|---|
| `GRADE_TIMEOUT` | `GRADE` tidak diterima dalam batas waktu setelah `DETECT` |
| `BUSY` | `GRADE` diterima saat tidak sedang menunggu grade, atau `id` tidak cocok |
| `BAD_CMD` | Pesan tidak dikenal atau format salah |
| `NO_ARRIVAL`, `JAM` | Cadangan, dipakai bila kelak ada sensor konfirmasi di wadah |

Setiap `ERR` dari fase gerak membuat firmware menghentikan konveyor dan menutup semua gate.

## Perilaku saat `GRADE_TIMEOUT`

Pusher tidak bergerak. Firmware mengirim `ERR,<id>,GRADE_TIMEOUT`, lalu menunggu sinar IR bebas (tomat diambil operator) sebelum mengirim `READY` dan mencari tomat lagi. Ini mencegah siklus ulang pada tomat yang sama.

## Mode debug manual

Saat menunggu grade, mengetik `A`, `B`, atau `C` (tanpa ID) di serial monitor tetap diterima untuk menguji servo. Pemetaan ke jalur fisik: `C` = jalur lurus, `B` = servo 2, `A` = servo 3. Perintah lama `busuk`/`setengah`/`matang` dihapus karena "busuk" tidak sama dengan Grade C (C juga mencakup tomat mentah).

## Batas waktu

Nilai batas waktu (menunggu grade, waktu tempuh tiap jalur) ada di konstanta firmware. Catat nilai finalnya di `docs/` setelah diukur.
