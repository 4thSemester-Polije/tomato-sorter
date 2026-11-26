# Dataset cacat v2 (AgML): persiapan, QA, dan training

Skrip `ml/prepare_defect_v2.py` mengubah dataset AgML (`Fresh/` dan `Rotten/`) menjadi dataset
klasifikasi 2 kelas yang siap dipakai `ml.train`: crop persegi per tomat, latar netral, dan
`split_manifest.csv` yang membagi data per kelompok agar tidak bocor.

Butuh patch 0005 (logging dan `--split-manifest`) sudah diterapkan.

## 1. Alur

```
AgML (Fresh, Rotten) --prepare_defect_v2--> defect_v2/{healthy,reject}, split_manifest.csv, qa/
        --QA visual (wajib)--> hapus crop yang salah label
        --ml.train --tasks defect ...--> model + logs/<run>/report.md (gate di foto primer)
```

## 2. Menjalankan persiapan (di PC, bukan Colab)

Dataset AgML ada di PC Anda, dan hasil crop kecil (ratusan KB per ribuan gambar), jadi jalankan di
terminal VS Code lalu unggah folder hasilnya ke Drive untuk training di Colab.

```powershell
.\.venv\Scripts\Activate.ps1
# uji cepat (40 gambar pertama per kelas) untuk melihat laporannya dulu
python -m ml.prepare_defect_v2 `
  --agml-dir "D:\D4 Polije\Semester 5\Project\Data\Datasets\Sekunder\agml" `
  --output-dir "D:\D4 Polije\Semester 5\Project\Data\Datasets\Sekunder\defect_v2_uji" --limit 40

# penuh
python -m ml.prepare_defect_v2 `
  --agml-dir "D:\D4 Polije\Semester 5\Project\Data\Datasets\Sekunder\agml" `
  --output-dir "D:\D4 Polije\Semester 5\Project\Data\Datasets\Sekunder\defect_v2"
```

| Opsi | Bawaan | Fungsi |
|---|---|---|
| `--background neutral\|keep` | `neutral` | `neutral`: latar diganti warna netral (meja, kertas, sabuk tidak bisa jadi petunjuk). `keep`: latar asli |
| `--multi-rotten exclude\|include` | `exclude` | Foto Rotten dengan beberapa tomat dikecualikan (label per buah tidak pasti) atau tetap dipakai |
| `--group-block N` | 20 | Foto bernomor berdekatan dianggap satu sesi pemotretan (satu kelompok); 0 = nonaktif |
| `--dup-threshold` | 0.985 | Korelasi minimum dua crop dianggap salinan; 0 = nonaktif. Jangan turun di bawah ~0.95 |
| `--size` | 256 | Sisi crop persegi |
| `--limit N` | - | Hanya N gambar pertama per kelas (uji cepat) |
| `--overwrite` | - | Timpa hasil sebelumnya (tanpa ini, ditolak) |

## 3. Apa yang disaring, dan kenapa

- **Tomat bersentuhan atau menyatu ditolak** (uji bentuk `solidity` < 0.92). Pemisahan otomatis
  (watershed) saya coba dan terbukti merusak bentuk tomat, jadi tidak dipakai.
- **Foto Rotten bermultitomat dikecualikan** secara bawaan. Foto Rotten sering berisi 2-3 tomat dan
  tidak semuanya busuk. Tanpa ini model bisa belajar "banyak tomat = reject".
- **Crop persegi** (bentuk tomat tidak terdistorsi), bukan kotak mentah yang diregangkan.
- **Kelompok split**: crop dari foto yang sama, foto bernomor berdekatan, dan crop hampir identik
  selalu masuk split yang sama. Gambar nyaris identik yang muncul di Fresh dan Rotten dibuang
  (`label_konflik`).

## 4. QA wajib sebelum training

1. Buka `prepare_report.md`:
   - **Gambar yang dikecualikan**: bila `multi_tomat_rotten` sangat besar (mis. lebih dari separuh
     Rotten), positif terlalu sedikit; pertimbangkan `--multi-rotten include` lalu tinjau `qa/`.
   - **Kelompok terbesar** lebih dari ~10% total crop berarti kelompok berantai; naikkan
     `--dup-threshold`.
   - **Split**: tiap kelas harus punya crop di train, validation, dan test.
2. Buka `qa/crops_healthy.jpg` dan `qa/crops_reject.jpg` (48 crop acak per kelas). Periksa:
   crop utuh, latar netral bersih, dan label masuk akal (mis. tomat sehat di folder `reject`).
3. Buka `qa/excluded_<alasan>.jpg` untuk memastikan yang dibuang memang layak dibuang. Tomat pucat
   atau latar yang tidak polos bisa gagal segmentasi.
4. Hapus file crop yang salah label dari `healthy/` atau `reject/`. Manifest tetap valid (entri
   tanpa file diabaikan saat training).

## 5. Training

```powershell
python -m ml.train --data-root "D:\D4 Polije\Semester 5\Project\Data\Datasets" `
  --tasks defect `
  --defect-data-dir "D:\D4 Polije\Semester 5\Project\Data\Datasets\Sekunder\defect_v2" `
  --split-manifest  "D:\D4 Polije\Semester 5\Project\Data\Datasets\Sekunder\defect_v2\split_manifest.csv" `
  --models-dir models_cacat_v2 --run-name defect_v2
```

Colab: lihat `docs/TRAINING_CACAT.md` (bagian 4) dan ganti dua path `--defect-data-dir` dan
`--split-manifest` ke folder `defect_v2` di Drive. Bila OpenCV belum ada di lingkungan training:
`pip install opencv-python-headless`.

**`--primary-crop`** (bawaan `auto`) menentukan masukan evaluasi foto primer. `auto` membaca
`prepare_report.json` dan memakai crop yang sama dengan cara dataset dibuat. Tanpa ini model yang
dilatih pada crop akan diuji dengan foto penuh dan gate-nya menyesatkan; bila Anda memaksa nilai
berbeda, training memberi peringatan. Foto primer yang tidak bisa di-crop (tidak ada tomat utuh)
tidak dihitung dan dilaporkan di kolom "tanpa crop".

### Bandingkan latar netral dan latar asli

Latar netral adalah hipotesis, bukan fakta. Buat dua dataset dan latih keduanya:

```powershell
python -m ml.prepare_defect_v2 --agml-dir ... --output-dir ...\defect_v2_keep --background keep
```

Latih dengan `--run-name defect_v2_keep` dan `--models-dir models_cacat_v2_keep`, lalu bandingkan
`report.md` kedua run di bagian "Primer". Pilih berdasarkan gate di foto primer, bukan akurasi
test sekunder.

## 6. Batasan yang perlu diketahui

- **Ambang belum dikalibrasi.** Segmentasi dan uji bentuk dicoba pada 18 foto (AgML dan primer),
  bukan seluruh dataset. `prepare_report.md` dan `qa/` adalah pemeriksaan sebenarnya.
- **Tomat yang sama difoto berkali-kali tidak terdeteksi.** Detektor kemiripan hanya menangkap
  salinan hampir identik. Di AgML, tomat yang sama tampak muncul di foto berbeda (mis. Rotten 0300
  dan 0600), sehingga akurasi validasi dan test sekunder cenderung optimistis. Foto primer adalah
  ukuran yang sebenarnya.
- **Cakupan cacat terbatas.** Rotten berisi busuk, keriput, dan bercak gelap. Retak dan lubang
  (mayoritas level 3 di foto primer Anda) kemungkinan besar tidak tercakup.
- **Model v2 menuntut crop yang sama saat dipakai.** API dan controller belum meng-crop tomat
  (patch berikutnya). Jangan pasang model v2 ke API sebelum itu, dan biarkan patch 0004 (grade dari
  kematangan saja) tetap aktif.
- **Di alat, crop memakai uji bentuk yang lebih longgar** (`min_solidity=0`), karena hanya ada satu
  tomat. Pada foto primer, tomat dengan kelopak besar (T003) tertolak oleh uji ketat tetapi
  lolos dengan yang longgar.
