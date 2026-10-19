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

## Keputusan tim: stasiun foto

Tomat ditaruh MANUAL oleh operator di stasiun foto, pada dudukan. Alasan: kondisi pengambilan gambar dapat diulang (posisi, jarak, cahaya sama setiap kali), bukan karena terlihat lebih profesional.

Syarat stasiun:
- Dudukan dangkal: cukup untuk mencegah tomat menggelinding, tetapi tidak menjebak sehingga pusher masih bisa mendorong tomat keluar.
- Webcam tetap di atas dudukan, jarak tidak berubah. Lampu LED difus dalam light box tertutup, latar polos kontras.
- IR mengarah ke dudukan.
- Orientasi tomat sama setiap kali (diputuskan tim, lihat `docs/SOP_OPERATOR.md`).

Konsekuensi: sistem ini SEMI-OTOMATIS (pemuatan tomat manual). Yang otomatis adalah inspeksi dan pemilahan. Tulis apa adanya di laporan.

Sinkronisasi dengan PC:
- IR memicu `DETECT`. PC menunggu tomat diam (`capture.settle_after_detect_s`) lalu memastikan ada tomat valid (mask tidak kosong dan luasnya wajar).
- Jika tidak ada tomat valid (pemicu palsu, mis. hanya tangan), PC tidak membalas dan firmware berakhir di `GRADE_TIMEOUT`.

Usulan firmware: konveyor MATI saat idle dan hanya menyala pada tahap `MENGANTAR`. Kode saat ini menyalakan konveyor saat idle (`CARI_BUAH`), yang tidak selaras dengan alur ini.

## Peran IR menurut kode

IR di kode berfungsi sebagai pendeteksi kehadiran tomat yang memicu "berhenti dan minta grade". Ini juga menjadi pemicu foto (protokol `DETECT`). IR tidak dipakai untuk mengonfirmasi tomat masuk wadah; akhir pengantaran murni berbasis waktu.

## Alur dengan PC (target)

`READY` -> IR terpicu -> `DETECT,<id>` -> PC menunggu tomat diam, memotret, menghitung grade -> `GRADE,<id>,<g>` -> `ACK` -> pusher -> pengantaran -> `DONE` -> `READY`.

## Perlu dikonfirmasi tim hardware

- Verifikasi fisik: IR mengarah ke dudukan di stasiun foto, dan tomat pada dudukan berada di tengah bidang pandang webcam.
- Konveyor berjalan saat idle di kode saat ini; usulan: mati saat idle. Konfirmasi.
- Urutan posisi gate servo 2 dan servo 3 sepanjang konveyor, dan apakah jalur lurus berakhir di wadah ketiga.
- Perlu tidaknya sensor konfirmasi di wadah.
