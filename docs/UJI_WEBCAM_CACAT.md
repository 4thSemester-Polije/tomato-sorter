# Uji webcam model cacat v2

Tujuan: mengukur model cacat v2 pada domain yang sebenarnya (webcam di stasiun foto), yang belum
pernah diuji. Foto primer dan AgML sama-sama foto HP, jadi hasil "LULUS" di sana belum membuktikan
apa pun untuk webcam.

Butuh patch 0005-0007 sudah diterapkan.

## 1. Model yang dipakai

Pakai **`defect_v2_ft`** (training dengan fine-tuning, seed 42). Alasan: ambang terbaiknya di foto
primer dekat 0,5, jadi ambang bawaan 0,5 bisa dipakai tanpa menyetelnya pada 9 foto primer, dan
recall rejectnya di validasi sedikit paling baik di antara run ft. Run seed 43-45 hanya
cadangan; perbedaannya kecil.

Model kematangan tetap yang lama. Dua model dari folder berbeda digabung di folder khusus API:

```powershell
cd "D:\D4 Polije\Semester 5\Project\Codingan\tomato-sorter"
mkdir models\api_v2
Copy-Item models\ripeness_resnext101.keras, models\ripeness_resnext101.json models\api_v2\
# model cacat v2 yang diunduh dari Drive (folder defect_v2_ft):
Copy-Item "<folder-unduhan>\defect_v2_ft\defect_resnext101.keras", "<folder-unduhan>\defect_v2_ft\defect_resnext101.json" models\api_v2\
```

Folder `models\api_v2` sudah tercakup `.gitignore` (`models/*`). Model cacat lama di `models\` tidak
tersentuh.

## 2. Menjalankan API

```powershell
$env:TOMATO_MODELS_DIR = "models\api_v2"
$env:TOMATO_GRADE_USES_DEFECT = "1"      # cacat ikut grade (untuk uji; tanpa ini hanya dicatat)
$env:TOMATO_DEFECT_THRESHOLD = "0.5"     # DIBEKUKAN sebelum uji; jangan diubah di tengah uji
python -m uvicorn ml.api:app --port 9000
```

Cek model mana yang aktif: buka `http://127.0.0.1:9000/health` dan lihat `defect_head`:

- `model_file` menunjuk `models\api_v2\defect_resnext101.keras`
- `trained_input` harus `crop_neutral` (bila `full_frame`, itu model cacat lama)
- `threshold` 0.5 dan `used_for_grade` true

`python -m tools.camera_predict` juga mencetak ringkasan ini saat mulai.

## 3. Mengambil foto uji

```powershell
python -m tools.camera_predict --no-dashboard --log-dir data\webcam_test
```

SPASI = foto dan klasifikasi. Tiap foto menyimpan frame mentah `raw_<waktu>.jpg` dan satu baris di
`data\webcam_test\results.csv`; terminal mencetak nomor urut tomat (`log uji: tomat no. N`).

Protokol:

- **30-40 tomat berbeda**, minimal **10 cacat jelas** (retak, lubang, keriput, bercak/busuk gelap)
  dan 20 tanpa cacat dengan variasi kematangan. Tambahan beberapa cacat ringan (level 2).
- Pencahayaan dan latar **sama dengan kondisi mesin yang sebenarnya**. Jangan menata ulang demi hasil.
- Letakkan tomat tanpa memilih sisi terbaik; sisi cacat sebaiknya sesekali menghadap atas, sesekali
  menyamping. Satu tomat, satu foto (foto ulang tomat sama tidak menambah bukti).
- Tulis nomor urut tomat (dari terminal) di kertas bersama keadaan tomatnya.

## 4. Pelabelan dan evaluasi

Buka `results.csv`, isi `true_defect_level` (**1** = tanpa cacat, **2** = cacat ringan,
**3** = cacat jelas) dan, bila perlu, `true_ripeness` dan `notes`. Lalu:

```powershell
python -m tools.eval_webcam_defect --csv data\webcam_test\results.csv --out data\webcam_test\laporan.md
```

Gerbang (ditetapkan sebelum uji): recall level 3 >= 70% dan level 1 salah ditandai <= 10% pada
ambang beku 0,5, dengan foto level 3 minimal 10. Foto yang tidak dinilai (crop gagal atau tomat tidak
valid) dihitung sebagai cacat yang terlewat. Laporan menampilkan rentang kepercayaan 95%; pada
n sekecil ini rentangnya lebar, jadi baca angkanya bersama rentang itu.

**Jangan memilih ambang dari tabel sapuan lalu menilainya pada data yang sama.** Bila ambang
ingin diubah, uji ambang baru pada foto webcam yang baru.

## 5. Bila ada masalah

| Gejala | Penyebab dan tindakan |
|---|---|
| `defect_status` = `tanpa_frame_mentah` | Klien tidak mengirim frame mentah: pastikan memakai `tools.camera_predict` dari patch ini dan API versi ini |
| `defect_status` = `crop_gagal` | Segmentasi tidak menemukan tomat utuh. Buka `raw_<waktu>.jpg`: biasanya latar tidak polos, tomat menyentuh tepi, atau pencahayaan ekstrem |
| `defect_status` = `tomat_tidak_valid` | Segmentasi pipeline PCV menolak gambar sebelum API dipanggil |
| `/health` menunjukkan `trained_input: full_frame` | Folder model salah: itu model cacat lama. Periksa `TOMATO_MODELS_DIR` |
| Banyak `crop_gagal` pada tomat pucat/hijau | Segmentasi berbasis saturasi lemah pada tomat pucat; catat jumlahnya, itu temuan untuk laporan |

## 6. Batas patch ini

- **Controller produksi (`controller/main.py`) belum mengirim frame mentah**, jadi di jalur mesin
  sebenarnya model cacat v2 tetap "tidak dinilai" dan grade mengikuti kematangan, sama seperti
  sebelumnya. Itu disengaja: dihubungkan setelah uji webcam ini lulus.
- Model kematangan tetap menerima gambar PCV seperti biasa; jalurnya tidak diubah.
