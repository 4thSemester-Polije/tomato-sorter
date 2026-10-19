# SOP Operator (draft)

1. Pastikan sistem menampilkan READY dan stasiun foto kosong.
2. Ambil satu tomat dan letakkan di dudukan dengan orientasi yang sama setiap kali (ditentukan tim, mis. kelopak menghadap atas).
3. Jauhkan tangan dari stasiun, lalu tunggu hasil. Jangan menaruh tomat berikutnya sebelum READY.
4. Bila muncul ERR: hentikan, ambil tomat dari stasiun/konveyor, catat kode ERR, tunggu READY.
5. Jangan menggeser webcam, lampu, light box, atau dudukan. Perubahan apa pun mewajibkan kalibrasi ulang (`mm_per_pixel`, ambang warna, foto latar kosong).
6. Saat pengumpulan data: satu ID tomat per tomat, foto dari beberapa posisi, dan catat di `data/photos.csv` (otomatis oleh `tools/capture_dataset.py`).
7. Tombol stop darurat memutus daya servo dan motor. Ketahui letaknya sebelum menyalakan alat.
