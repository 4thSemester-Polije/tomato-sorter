"""Ekstraksi fitur warna kematangan tomat menggunakan ruang warna HSV (tanpa keputusan grade).

Rentang warna kematangan (skala Hue OpenCV 0-179):
- Merah  (matang) : H 0..10 atau 170..180
- Oranye (sedang) : H 11..22
- Kuning (mentah) : H 23..35
- Hijau  (mentah) : H 36..85
"""
from __future__ import annotations

import cv2
import numpy as np

HSV_MERAH1_LOWER = np.array([0, 70, 50], dtype=np.uint8)
HSV_MERAH1_UPPER = np.array([10, 255, 255], dtype=np.uint8)
HSV_MERAH2_LOWER = np.array([170, 70, 50], dtype=np.uint8)
HSV_MERAH2_UPPER = np.array([180, 255, 255], dtype=np.uint8)

HSV_ORANYE_LOWER = np.array([11, 70, 50], dtype=np.uint8)
HSV_ORANYE_UPPER = np.array([22, 255, 255], dtype=np.uint8)

HSV_KUNING_LOWER = np.array([23, 60, 50], dtype=np.uint8)
HSV_KUNING_UPPER = np.array([35, 255, 255], dtype=np.uint8)

HSV_HIJAU_LOWER = np.array([36, 40, 40], dtype=np.uint8)
HSV_HIJAU_UPPER = np.array([85, 255, 255], dtype=np.uint8)


def extract_hsv_color_features(img_bgr, mask) -> dict:
    """Hitung persentase warna kematangan HSV (merah, oranye, kuning, hijau) di dalam area mask tomat.

    Input:
        img_bgr: Citra tomat BGR (misalnya hasil crop 1:1).
        mask: Masker biner uint8 (0/255) dengan ukuran sama dengan img_bgr.
    Output:
        dict berisi total_masked_pixels, persen_merah, persen_oranye, persen_kuning, persen_hijau,
        serta masker masing-masing warna (tanpa logika tingkat/grade).
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    total_px = int(np.sum(mask > 0))

    m_merah1 = cv2.inRange(img_hsv, HSV_MERAH1_LOWER, HSV_MERAH1_UPPER)
    m_merah2 = cv2.inRange(img_hsv, HSV_MERAH2_LOWER, HSV_MERAH2_UPPER)
    mask_merah = cv2.bitwise_and(cv2.bitwise_or(m_merah1, m_merah2), mask)

    mask_oranye = cv2.bitwise_and(cv2.inRange(img_hsv, HSV_ORANYE_LOWER, HSV_ORANYE_UPPER), mask)
    mask_kuning = cv2.bitwise_and(cv2.inRange(img_hsv, HSV_KUNING_LOWER, HSV_KUNING_UPPER), mask)
    mask_hijau = cv2.bitwise_and(cv2.inRange(img_hsv, HSV_HIJAU_LOWER, HSV_HIJAU_UPPER), mask)

    if total_px > 0:
        persen_merah = float((np.sum(mask_merah > 0) / total_px) * 100.0)
        persen_oranye = float((np.sum(mask_oranye > 0) / total_px) * 100.0)
        persen_kuning = float((np.sum(mask_kuning > 0) / total_px) * 100.0)
        persen_hijau = float((np.sum(mask_hijau > 0) / total_px) * 100.0)
    else:
        persen_merah = persen_oranye = persen_kuning = persen_hijau = 0.0

    return {
        "total_masked_pixels": total_px,
        "persen_merah": persen_merah,
        "persen_oranye": persen_oranye,
        "persen_kuning": persen_kuning,
        "persen_hijau": persen_hijau,
        "mask_merah": mask_merah,
        "mask_oranye": mask_oranye,
        "mask_kuning": mask_kuning,
        "mask_hijau": mask_hijau,
    }

