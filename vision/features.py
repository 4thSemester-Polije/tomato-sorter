"""Ekstraksi fitur. NAMA FITUR ADALAH KONTRAK ANTAR MODUL: jangan diubah tanpa memberi tahu tim.

Semua ambang warna di bawah adalah nilai awal dan BELUM DIKALIBRASI. Setel dengan foto dari alat sendiri
(lihat gambar debug di data/debug/).
"""
import math

import cv2
import numpy as np

FEATURE_NAMES = (
    "pct_red",      # persen piksel merah di dalam area tomat (0-100), untuk kematangan
    "mean_h",       # rata-rata Hue (sirkular, 0-179, skala OpenCV)
    "mean_s",       # rata-rata Saturation (0-255)
    "mean_v",       # rata-rata Value (0-255)
    "pct_dark",     # persen piksel gelap di dalam area tomat (0-100), proksi cacat
    "diameter_px",  # diameter setara dari luas (piksel)
    "diameter_mm",  # diameter dalam mm; None bila mm_per_pixel belum dikalibrasi
)

RED_HUE_LOW = 10      # merah di HSV OpenCV membungkus: H <= 10 atau H >= 170
RED_HUE_HIGH = 170
SAT_MIN = 90          # saturasi minimum agar dihitung merah
VAL_MIN = 70          # di bawah ini dianggap gelap (bukan merah)
ERODE_PX = 5          # kikis mask untuk statistik cacat (hindari bayangan tepi)


def _circular_mean_hue(h_values):
    """Rata-rata hue sirkular (0-179). Rata-rata biasa salah untuk merah (hue 2 dan 177 akan menjadi ~90)."""
    ang = h_values.astype(np.float64) * math.pi / 90.0  # 0..179 -> 0..2*pi
    mean_ang = math.atan2(np.sin(ang).mean(), np.cos(ang).mean())
    return (math.degrees(mean_ang) % 360.0) / 2.0


def extract_features(image, mask, mm_per_pixel=None) -> dict:
    """Input: gambar BGR dan mask (uint8) dari segment(). Output: dict dengan kunci FEATURE_NAMES.

    Melempar ValueError bila mask kosong.
    """
    inside = mask > 0
    area = int(inside.sum())
    if area == 0:
        raise ValueError("Mask kosong: tomat tidak ditemukan")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    red = ((h <= RED_HUE_LOW) | (h >= RED_HUE_HIGH)) & (s >= SAT_MIN) & (v >= VAL_MIN)
    pct_red = 100.0 * float((red & inside).sum()) / area

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ERODE_PX + 1, 2 * ERODE_PX + 1))
    inner = cv2.erode(mask, kernel) > 0
    if not inner.any():
        inner = inside
    pct_dark = 100.0 * float(((v < VAL_MIN) & inner).sum()) / float(inner.sum())

    diameter_px = math.sqrt(4.0 * area / math.pi)
    return {
        "pct_red": pct_red,
        "mean_h": _circular_mean_hue(h[inside]),
        "mean_s": float(s[inside].mean()),
        "mean_v": float(v[inside].mean()),
        "pct_dark": pct_dark,
        "diameter_px": diameter_px,
        "diameter_mm": diameter_px * mm_per_pixel if mm_per_pixel else None,
    }
