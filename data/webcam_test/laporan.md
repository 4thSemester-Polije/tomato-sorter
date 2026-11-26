# Evaluasi uji webcam model cacat

## Per level cacat sebenarnya

| level   |   n | tidak dinilai | ditandai cacat | proporsi                   |
|---------|----:|--------------:|---------------:|----------------------------|
| level 1 |   3 |             0 |              0 | 0/3 = 0% (95%: 0%-56%)     |
| level 2 |   0 |             0 |              0 | -                          |
| level 3 |   3 |             0 |              3 | 3/3 = 100% (95%: 44%-100%) |

## Ringkasan

| ukuran                                           | nilai                      |
|--------------------------------------------------|----------------------------|
| ambang (dibekukan)                               | 0.5000                     |
| baris di CSV / berlabel                          | 10 / 6                     |
| recall level 3 (tidak dinilai dihitung terlewat) | 3/3 = 100% (95%: 44%-100%) |
| level 1 salah ditandai cacat                     | 0/3 = 0% (95%: 0%-56%)     |
| AUC level 1 vs 3 (hanya yang dinilai)            | 1.0000                     |

## Gerbang

Syarat (ambang beku 0.5): recall level 3 >= 70%, level 1 salah ditandai <= 10%, dan foto level 3 minimal 10.
Hasil: **TIDAK DAPAT DINILAI (foto level 3 kurang dari 10, atau tanpa level 1)**.

## Sapuan ambang (hanya informasi)

| ambang | level 3 ditandai           | level 1 ditandai       | level 2 ditandai |
|-------:|----------------------------|------------------------|------------------|
| 0.2000 | 3/3 = 100% (95%: 44%-100%) | 0/3 = 0% (95%: 0%-56%) | -                |
| 0.3000 | 3/3 = 100% (95%: 44%-100%) | 0/3 = 0% (95%: 0%-56%) | -                |
| 0.4000 | 3/3 = 100% (95%: 44%-100%) | 0/3 = 0% (95%: 0%-56%) | -                |
| 0.5000 | 3/3 = 100% (95%: 44%-100%) | 0/3 = 0% (95%: 0%-56%) | -                |
| 0.6000 | 3/3 = 100% (95%: 44%-100%) | 0/3 = 0% (95%: 0%-56%) | -                |
| 0.7000 | 3/3 = 100% (95%: 44%-100%) | 0/3 = 0% (95%: 0%-56%) | -                |
| 0.8000 | 2/3 = 67% (95%: 21%-94%)   | 0/3 = 0% (95%: 0%-56%) | -                |

Jangan memilih ambang dari tabel ini lalu menilainya pada data yang sama; ambang baru harus diuji pada foto webcam baru.
