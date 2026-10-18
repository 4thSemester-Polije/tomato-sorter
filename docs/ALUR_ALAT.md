# Alur Kerja Alat (sesuai rancangan tim hardware)

Status: draft, menunggu penjelasan posisi sensor IR dan servo 2 relatif terhadap tiga wadah.

1. Tomat ditaruh manual di stasiun foto (terpisah dari conveyor), pada dudukan kecil agar tidak menggelinding.
2. PC mendeteksi tomat hadir (vision) dan mengambil gambar.
3. PC menghitung grade, lalu mengirim `GRADE`.
4. Servo 1 menendang tomat ke landasan conveyor (hanya SETELAH grade diterima).
5. Conveyor berjalan. Servo 2 menendang sesuai grade.
6. Sensor IR mengonfirmasi tomat masuk lubang, conveyor berhenti, Arduino melapor `DONE`.

## State machine Arduino (usulan)

`READY -> [terima GRADE] -> KICK1 -> RUN -> ARRIVE -> KICK2 -> IN_BIN -> STOP -> DONE -> READY`

Setiap tahap menunggu punya batas waktu; bila terlewati pindah ke `ERROR` (conveyor berhenti, kirim `ERR`).

## Keputusan yang masih terbuka

- Posisi sensor IR dan servo 2 relatif terhadap tiga wadah.
- Perlu tidaknya sensor IR kedua (sensor 1: tomat tiba di servo 2, sensor 2: konfirmasi masuk lubang).
- Pemicu foto: deteksi otomatis oleh vision (usulan) dengan cadangan tombol/keyboard.
