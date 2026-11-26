"""Crop satu tomat dari foto: deteksi kandidat, uji bentuk, crop persegi, latar netral.

Dipakai bersama oleh skrip persiapan dataset (ml/prepare_defect_v2.py) dan, nanti, saat
inferensi, supaya data training dan data nyata diproses dengan cara yang identik.
Ambang di sini adalah nilai awal dan BELUM DIKALIBRASI pada webcam alat.

Asumsi gambar: BGR, sisi terpanjang sekitar 400-800 px (kernel morfologi bernilai piksel tetap).
Tomat yang saling bersentuhan sengaja TIDAK dipisahkan (pemisahan watershed terbukti merusak
bentuk); gabungannya ditolak lewat uji solidity pada is_clean_single_tomato.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vision.segmentation import _clean

# Rata-rata ImageNet (RGB 124,116,104) dalam urutan BGR: terlihat "netral" bagi model ImageNet.
NEUTRAL_BACKGROUND_BGR = (104, 116, 124)

MAX_SIDE = 800  # sisi terpanjang gambar yang diproses (kernel morfologi bernilai piksel tetap)
MIN_AREA_FRACTION = 0.01
MAX_AREA_FRACTION = 0.90
MIN_SOLIDITY = 0.92  # tomat tunggal ~0,95-0,99; dua tomat menyatu ~0,5-0,9 (sampel AgML)


@dataclass(frozen=True)
class TomatoRegion:
    box: tuple[int, int, int, int]  # x, y, lebar, tinggi
    mask: np.ndarray  # uint8 0/255, ukuran sama dengan gambar, kontur terisi penuh
    area_fraction: float
    solidity: float
    touches_border: bool


def pil_to_bgr(image, max_side: int = MAX_SIDE) -> np.ndarray:
    """Gambar PIL -> BGR, diperkecil bila sisi terpanjangnya melebihi max_side."""
    from PIL import Image

    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        rgb = rgb.resize((round(width * scale), round(height * scale)), Image.LANCZOS)
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


def _saturation_binary(image_bgr: np.ndarray) -> np.ndarray:
    saturation = cv2.GaussianBlur(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[..., 1], (5, 5), 0)
    _, binary = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _clean(binary)


def find_tomato_regions(
    image_bgr: np.ndarray,
    min_area_fraction: float = MIN_AREA_FRACTION,
    max_area_fraction: float = MAX_AREA_FRACTION,
) -> list[TomatoRegion]:
    """Semua kandidat tomat (komponen terhubung hasil segmentasi saturasi), terbesar dulu."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Gambar input kosong atau tidak valid")
    height, width = image_bgr.shape[:2]
    contours, _ = cv2.findContours(
        _saturation_binary(image_bgr), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    regions: list[TomatoRegion] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        fraction = area / float(height * width)
        if not (min_area_fraction <= fraction <= max_area_fraction):
            continue
        filled = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        x, y, w, h = cv2.boundingRect(contour)
        regions.append(
            TomatoRegion(
                box=(x, y, w, h),
                mask=filled,
                area_fraction=float(fraction),
                solidity=float(area / hull_area) if hull_area > 0 else 0.0,
                touches_border=bool(x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1),
            )
        )
    return sorted(regions, key=lambda region: region.area_fraction, reverse=True)


def is_clean_single_tomato(region: TomatoRegion, min_solidity: float = MIN_SOLIDITY) -> bool:
    """Utuh (tidak menyentuh tepi gambar) dan cembung (bukan gabungan beberapa tomat)."""
    return (not region.touches_border) and region.solidity >= min_solidity


def crop_region(
    image_bgr: np.ndarray,
    region: TomatoRegion,
    padding: float = 0.12,
    size: int = 256,
    neutral_background: bool = True,
) -> np.ndarray:
    """Crop persegi (bentuk tomat tidak terdistorsi) berukuran size x size.

    neutral_background=True: piksel di luar tomat (dengan tepi diperlebar sedikit agar cacat di
    pinggir tidak terpotong) diganti warna netral, sehingga model tidak bisa memakai latar
    (meja kayu, kertas, sabuk) sebagai petunjuk.
    """
    height, width = image_bgr.shape[:2]
    x, y, w, h = region.box
    side = int(round(max(w, h) * (1.0 + 2.0 * padding)))
    side = max(8, min(side, height, width))
    center_x, center_y = x + w // 2, y + h // 2
    left = int(np.clip(center_x - side // 2, 0, width - side))
    top = int(np.clip(center_y - side // 2, 0, height - side))

    source = image_bgr
    if neutral_background:
        grow = max(2, int(0.025 * max(w, h)))
        keep = cv2.dilate(
            region.mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        )
        source = image_bgr.copy()
        source[keep == 0] = NEUTRAL_BACKGROUND_BGR
    crop = source[top : top + side, left : left + side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def crop_largest_tomato(
    image_bgr: np.ndarray,
    size: int = 256,
    neutral_background: bool = True,
    min_solidity: float = MIN_SOLIDITY,
) -> np.ndarray | None:
    """Untuk inferensi: crop tomat terbesar yang utuh (tidak menyentuh tepi), atau None.

    Saat membangun dataset, min_solidity bawaan menjaga kemurnian (tomat menyatu ditolak). Di alat
    nyata hanya ada satu tomat, jadi min_solidity=0.0 lebih tepat: tomat dengan kelopak besar
    (solidity ~0,89 pada foto primer T003) tetap di-crop.
    """
    for region in find_tomato_regions(image_bgr):
        if is_clean_single_tomato(region, min_solidity):
            return crop_region(image_bgr, region, size=size, neutral_background=neutral_background)
    return None


def crop_for_inference(image, background: str, size: int = 256):
    """Gambar PIL -> crop PIL untuk model cacat, atau None bila tidak ada tomat utuh.

    Satu-satunya jalur crop saat inferensi dan evaluasi (API dan ml.train memakainya). background:
    "neutral" atau "keep", sama dengan cara dataset training dibuat. Uji solidity dilonggarkan
    (min_solidity=0): di alat hanya ada satu tomat, dan tomat berkelopak besar tidak boleh ditolak.
    """
    from PIL import Image

    if background not in ("neutral", "keep"):
        raise ValueError("background must be 'neutral' or 'keep'.")
    crop = crop_largest_tomato(
        pil_to_bgr(image), size=size, neutral_background=(background == "neutral"), min_solidity=0.0
    )
    return None if crop is None else Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
