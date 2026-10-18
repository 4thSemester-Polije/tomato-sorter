"""Segmentasi tomat dari latar. SEMUA ambang di sini adalah nilai awal dan BELUM DIKALIBRASI.

Dua cara:
- segment_by_background: bandingkan dengan foto stasiun kosong (disarankan; kamera dan latar tetap).
- segment_by_saturation: cadangan untuk latar polos berwarna pucat (putih/abu).
"""
import cv2
import numpy as np

MIN_AREA_PX = 500          # luas minimum objek (piksel), kalibrasi
MAX_AREA_FRAC = 0.9        # objek yang menutupi > 90% gambar dianggap gagal
MIN_DIFF_PEAK = 25.0       # selisih warna minimum agar dianggap ada objek (skala Lab)
L_WEIGHT = 0.5             # bobot kecerahan; kecil agar bayangan tidak dominan


def _clean(binary):
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    b = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
    return cv2.morphologyEx(b, cv2.MORPH_CLOSE, k_close)


def _largest_filled(binary, min_area_px):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(binary)
    if not contours:
        return out
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area_px or area > MAX_AREA_FRAC * binary.size:
        return out
    cv2.drawContours(out, [c], -1, 255, thickness=cv2.FILLED)
    return out


def segment_by_background(image, background, min_area_px=MIN_AREA_PX):
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(background, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = img_lab - bg_lab
    dist = np.sqrt(L_WEIGHT * d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
    if dist.max() < MIN_DIFF_PEAK:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    dist8 = cv2.GaussianBlur(np.clip(dist, 0, 255).astype(np.uint8), (5, 5), 0)
    _, binary = cv2.threshold(dist8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _largest_filled(_clean(binary), min_area_px)


def segment_by_saturation(image, min_area_px=MIN_AREA_PX):
    s = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 1], (5, 5), 0)
    _, binary = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _largest_filled(_clean(binary), min_area_px)


def segment(image, background=None, min_area_px=MIN_AREA_PX):
    """Input: gambar BGR (+ foto latar kosong berukuran sama, opsional).
    Output: mask uint8 (0/255). Mask kosong (semua 0) bila tomat tidak ditemukan.

    Catatan: warna latar/sabuk jangan merah, hijau, atau coklat. Waspadai kelopak hijau dan bayangan tepi.
    """
    if background is not None:
        if background.shape != image.shape:
            raise ValueError("Ukuran foto latar dan gambar harus sama")
        return segment_by_background(image, background, min_area_px)
    return segment_by_saturation(image, min_area_px)
