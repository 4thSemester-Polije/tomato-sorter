# Training model cacat: log, tabel metrik, dan cara menjalankan

Dokumen ini menjelaskan cara melatih model cacat (dan model lain) dengan pencatatan lengkap,
lewat terminal VS Code maupun Google Colab. Dataset cacat versi baru dan skrip persiapannya
datang di patch berikutnya; sampai saat itu Anda bisa menguji pencatatan dengan dataset lama
(lihat "Uji cepat").

## 1. Apa yang dicatat

Setiap run membuat satu folder `logs/<run-name>_<arsitektur>_<waktu>/`:

| File | Isi |
|---|---|
| `train.log` | Log teks lengkap (juga tampil di layar): jumlah gambar per kelas, tiap epoch, peringatan, error |
| `history.csv` | Satu baris per epoch: fase (`head`/`fine_tune`), loss, acc, val_loss, val_acc, learning rate, detik. Ditulis dan di-flush **tiap epoch**, jadi aman walau Colab terputus |
| `config.json` | Hyperparameter, seed, versi Python/TF/Keras, GPU/CPU, hash commit git |
| `report.md` | Semua tabel: epoch, jumlah data per split, metrik per kelas, confusion matrix (validasi, test, primer) |
| `metrics.json` | Versi mesin dari semua metrik |
| `predictions_primer.csv` | Hanya `--tasks defect`: label, prediksi, dan `p_reject` tiap foto primer |
| `curves_<task>.png`, `confusion_test_<task>.png` | Kurva loss/akurasi dan confusion matrix (butuh `matplotlib`, dilewati bila tidak ada) |
| `tensorboard/` | Hanya bila memakai `--tensorboard` |

Folder `logs/` sudah masuk `.gitignore`.

## 2. Opsi CLI

```
python -m ml.train --data-root <folder Datasets> [opsi]

--tasks ripeness defect     head yang dilatih (bawaan: keduanya)
--defect-data-dir <folder>  dataset cacat baru: berisi subfolder healthy/ dan reject/
--split-manifest <csv>      split per grup (kolom: file,split,group); untuk satu task per run
--models-dir <folder>       tempat model .keras dan .json
--logs-dir <folder>         folder induk log (bawaan: logs)
--run-name <nama>           awalan nama folder run
--tensorboard               tulis juga log TensorBoard
--overwrite-models          izinkan menimpa model lama saat melatih sebagian task
--head-epochs / --fine-tune-epochs / --batch-size / --seed
```

Pengaman: melatih sebagian task (misalnya hanya `defect`) ke folder yang sudah berisi model
akan **ditolak** kecuali memakai `--models-dir` lain atau `--overwrite-models`. Pakai
`--models-dir models_cacat_v2` agar model cacat lama tetap aman.

Split manifest: semua gambar yang berasal dari sumber yang sama (satu foto asli, satu video)
harus punya nilai `group` yang sama. Training akan berhenti dengan pesan jelas bila satu grup
terbelah ke lebih dari satu split (kebocoran data) atau ada gambar yang tidak tercantum.

## 3. Lewat terminal VS Code (Windows PowerShell)

TensorFlow 2.15.1 di Windows native hanya berjalan di CPU, jadi VS Code cocok untuk uji cepat
dan evaluasi. Training penuh ResNeXt-101 sebaiknya di Colab GPU.

1. Buka folder `tomato-sorter`, lalu buka terminal (`Ctrl+` `).
2. Aktifkan venv: `.\.venv\Scripts\Activate.ps1`
   (bila diblokir: `Set-ExecutionPolicy -Scope Process Bypass`).
3. Cek versi: `python --version` (harus 3.11) dan
   `python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"`.
4. Opsional untuk grafik dan TensorBoard: `pip install matplotlib tensorboard`.
5. **Uji cepat** (1 epoch, dataset lama, model ke folder terpisah):
   ```powershell
   python -m ml.train --data-root "D:\D4 Polije\Semester 5\Project\Data\Datasets" `
     --tasks defect --models-dir models_smoke --head-epochs 1 --fine-tune-epochs 0 --run-name smoke
   ```
6. Pantau log dari terminal kedua (ganti nama folder run):
   `Get-Content logs\smoke_resnext101_<waktu>\train.log -Wait -Tail 20`
7. TensorBoard (bila memakai `--tensorboard`): `tensorboard --logdir logs`, buka `http://localhost:6006`.
8. Setelah selesai, buka `logs\<run>\report.md` (pratinjau Markdown VS Code: `Ctrl+Shift+V`).

## 4. Lewat Google Colab

1. Di PC, taruh di Google Drive: zip repo (tanpa `.venv`, `models`, `data\debug`) dan folder
   `Datasets` (berisi `Primer` dan dataset yang dipakai).
2. Colab: `Runtime > Change runtime type > T4 GPU`, lalu `!nvidia-smi`.
3. Mount Drive:
   ```python
   from google.colab import drive; drive.mount('/content/drive')
   ```
4. Salin ke disk lokal Colab (lebih cepat dari Drive) dan unzip repo:
   ```python
   !cp /content/drive/MyDrive/tomato-sorter.zip /content/ && cd /content && unzip -q tomato-sorter.zip
   %cd /content/tomato-sorter
   ```
5. Cek `!python --version`. `requirements-ml.txt` memakai TensorFlow 2.15.1 yang butuh
   Python 3.11 (tidak ada paket untuk 3.12). Pakai runtime Colab dengan Python 3.11 seperti saat
   training sebelumnya.
6. Install: `%pip install -r requirements-ml.txt matplotlib`
7. Jalankan, dengan model dan log di Drive supaya tidak hilang saat runtime reset:
   ```python
   !python -m ml.train \
     --data-root "/content/drive/MyDrive/Datasets" \
     --tasks defect \
     --defect-data-dir "/content/drive/MyDrive/Datasets/Sekunder/defect_v2" \
     --split-manifest "/content/drive/MyDrive/Datasets/Sekunder/defect_v2/split_manifest.csv" \
     --models-dir "/content/drive/MyDrive/tomato-models/defect_v2" \
     --logs-dir "/content/drive/MyDrive/tomato-logs" \
     --run-name defect_v2
   ```
8. Pantau: `!tail -n 20 /content/drive/MyDrive/tomato-logs/<run>/train.log`, atau
   `%load_ext tensorboard` lalu `%tensorboard --logdir /content/drive/MyDrive/tomato-logs`
   (tambahkan `--tensorboard` saat training). Biarkan tab terbuka; Colab gratis memutus sesi idle.
9. Bila sesi terputus, `history.csv` dan `train.log` yang sudah tertulis tetap ada di Drive.
   Model terbaik tersimpan otomatis (`ModelCheckpoint`), tetapi training harus dimulai ulang.
10. Unduh `defect_resnext101.keras` + `.json` dan folder run dari Drive ke repo lokal.

## 5. Membaca laporan

Untuk `--tasks defect`, `report.md` memuat tabel berikut (urutan di file):

- **Epoch**: loss/acc/val_loss/val_acc per epoch. Val_loss yang naik sementara loss turun
  berarti overfit; keduanya datar di akurasi rendah berarti underfit.
- **Data**: jumlah gambar per kelas di tiap split.
- **Validasi / Test sekunder**: precision, recall, F1, confusion matrix. Angka ini hanya
  mengukur data sekunder; jangan dijadikan bukti model siap pakai.
- **Primer - model cacat per level**: dari foto primer, berapa yang diprediksi `reject` untuk
  tiap level cacat sebenarnya, plus rata-rata `p_reject`.
- **Primer - ringkasan**: AUC dan recall level 3 pada false positive <= 10%.
- **Primer - gate go/no-go**: lulus bila recall level 3 >= 70% pada false positive <= 10%
  dan foto level 3 minimal 5. Ambang dipilih dari data yang sama sehingga hasilnya optimistis;
  tambahkan foto webcam sebagai uji akhir sebelum model dipakai.

## 6. Masalah umum

- `FileExistsError: Model sudah ada`: pakai `--models-dir` lain atau `--overwrite-models`.
- `matplotlib tidak terpasang`: grafik dilewati; `pip install matplotlib`.
- `TensorBoard tidak terpasang`: `--tensorboard` dilewati; `pip install tensorboard`.
- Bagian "Primer - evaluasi gagal" di laporan: model tetap tersimpan, tetapi folder `Primer`
  (foto HEIC, `Label\Labelv.1.xlsx`, `mapping_nama_file.csv`) tidak ditemukan atau
  `pillow-heif`/`openpyxl` belum terpasang. Perbaiki lalu evaluasi ulang.
