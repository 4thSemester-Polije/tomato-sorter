# Rubrik Penentuan Grade Tomat v1.1

Objek: tomat sayur, difoto dari atas oleh satu webcam tetap.
Dokumen lengkap (dengan prosedur pelabelan dan contoh) ada di dokumen bersama tim. Ini salinan ringkas untuk repo.

## Dua atribut penentu grade

| Tingkat | Kematangan (acuan warna USDA, sisi atas) | Cacat terlihat dari atas |
|---|---|---|
| 1 | Merah >= 60% permukaan terlihat (Light Red/Red) | Tidak ada, atau bercak sangat kecil |
| 2 | Merah 10% sampai < 60% (Turning/Pink) | Ringan: bercak gelap atau lubang kecil, total < ~10% permukaan terlihat |
| 3 | Merah < 10% (Green/Breaker) | Berat: >= ~10%, busuk, atau pecah |

## Aturan

Grade = tingkat TERBURUK dari kedua atribut.

- A: kedua atribut tingkat 1
- B: tingkat terburuk adalah 2
- C: tingkat terburuk adalah 3 (belum layak jual segar saat ini: mentah atau cacat berat)

## Catatan

- Angka ~10% untuk cacat adalah USULAN tim, bukan standar. Kalibrasi dari foto nyata.
- Ukuran (diameter, mm) diukur sebagai fitur informasi, tidak menentukan grade.
- Kekerasan/kelembekan tidak dapat diukur kamera dan tidak termasuk rubrik.
- Keterbatasan: kamera tunggal dari atas, cacat dan warna di sisi bawah tidak terlihat.
- Gerbang evaluasi: dua minggu setelah foto pertama dari alat terkumpul, evaluasi atribut cacat. Bila tidak stabil, turunkan ke kematangan saja.
- Perubahan angka setelah pelabelan dimulai mengharuskan pelabelan ulang. Catat di riwayat versi dokumen bersama.
