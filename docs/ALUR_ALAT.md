# Alur Kerja Alat (disusun dari kode firmware ESP32 yang ada)

Status: sebagian perlu dikonfirmasi tim hardware (lihat bagian akhir).

## Alur menurut kode

1. `CARI_BUAH`: konveyor berjalan. IR menunggu buah (debounce 50 ms, cooldown 500 ms setelah siklus).
2. IR terpicu: konveyor BERHENTI, lalu firmware menunggu grade (saat ini diketik manual di serial monitor).
3. `MENYODOK`: servo pusher membuka 90 derajat selama 1 detik, lalu kembali.
4. `MENGANTAR`: konveyor berjalan, gate sesuai grade dibuka, selama waktu tetap (4 s untuk lurus, 4 s servo 2, 5 s servo 3).
5. Gate ditutup, penghitung bertambah, kembali ke langkah 1.

## Pemetaan grade ke jalur fisik

| Grade rubrik | Nama lama di firmware | Jalur |
|---|---|---|
| A | matang | Gate servo 3 |
| B | setengah | Gate servo 2 |
| C | busuk | Jalur lurus (tanpa gate) |

## Peran IR menurut kode

IR di kode berfungsi sebagai pendeteksi kehadiran tomat yang memicu "berhenti dan minta grade". Ini juga menjadi pemicu foto (protokol `DETECT`). IR tidak dipakai untuk mengonfirmasi tomat masuk wadah; akhir pengantaran murni berbasis waktu.

## Alur dengan PC (target)

`READY` -> IR terpicu -> `DETECT,<id>` -> PC menunggu tomat diam, memotret, menghitung grade -> `GRADE,<id>,<g>` -> `ACK` -> pusher -> pengantaran -> `DONE` -> `READY`.

## Perlu dikonfirmasi tim hardware

- Posisi fisik IR: apakah saat IR terpicu dan konveyor berhenti, tomat tepat berada di stasiun foto (di bawah webcam)?
- Tujuan konveyor berjalan saat menunggu tomat.
- Urutan posisi gate servo 2 dan servo 3 sepanjang konveyor, dan apakah jalur lurus berakhir di wadah ketiga.
- Perlu tidaknya sensor konfirmasi di wadah.
