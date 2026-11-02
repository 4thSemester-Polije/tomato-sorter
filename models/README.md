# models/

Model Keras hasil training per arsitektur (`<tugas>_<arsitektur>.keras`, dengan
tugas `ripeness`/`defect` dan arsitektur `resnext50`/`resnext101`), metadata JSON,
dan laporan `evaluation_<arsitektur>.json`. Model tidak di-commit secara default
(lihat `.gitignore`). Model final untuk demo boleh di-commit manual dengan
`git add -f models/<nama>`.

Prosedur training, evaluasi, perbandingan, dan API ada di `docs/ML_RESNEXT101.md`.
