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


def extract_ripeness_color_pct(img_bgr, mask_clean) -> dict:
    """FITUR 1 PCV: Hitung persentase warna dominan (Merah, Kuning/Oranye, Hijau) di dalam mask.

    Hanya mengekstraksi fitur persentase warna tanpa menentukan tingkat kematangan atau grade.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    active = mask_clean > 0
    pixels_r = img_rgb[:, :, 0][active]
    pixels_g = img_rgb[:, :, 1][active]
    pixels_b = img_rgb[:, :, 2][active]

    total_masked_pixels = int(np.sum(active))
    if total_masked_pixels > 0:
        red_dominant = int(np.sum((pixels_r > pixels_g) & (pixels_r > pixels_b)))
        green_dominant = int(np.sum((pixels_g > pixels_r) & (pixels_g > pixels_b)))
        yellow_orange_dominant = total_masked_pixels - (red_dominant + green_dominant)

        persen_merah = float((red_dominant / total_masked_pixels) * 100.0)
        persen_hijau = float((green_dominant / total_masked_pixels) * 100.0)
        persen_kuning = float(max(0.0, (yellow_orange_dominant / total_masked_pixels) * 100.0))
    else:
        persen_merah, persen_hijau, persen_kuning = 0.0, 0.0, 0.0

    return {
        "total_masked_pixels": total_masked_pixels,
        "persen_merah": persen_merah,
        "persen_kuning": persen_kuning,
        "persen_hijau": persen_hijau,
    }


def extract_defect_area(img_bgr, mask_clean, dark_threshold: int = 50, erode_size: int = 15) -> dict:
    """FITUR 2 PCV: Deteksi kontur utama, erosi ring pinggir, dan hitung persentase area cacat/bercak.

    Hanya mengekstraksi fitur area cacat dan maskernya tanpa menentukan tingkat cacat atau grade.
    """
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    luas_piksel = 0.0
    luas_evaluasi = 0
    total_cacat_pixels = 0
    persentase_cacat = 0.0
    cacat_mask = np.zeros_like(mask_clean, dtype=np.uint8)
    inner_contour_mask = np.zeros_like(mask_clean, dtype=np.uint8)
    main_contour = None

    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        luas_piksel = float(cv2.contourArea(main_contour))

        contour_mask = np.zeros_like(mask_clean, dtype=np.uint8)
        cv2.drawContours(contour_mask, [main_contour], -1, 255, -1)

        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
        inner_contour_mask = cv2.erode(contour_mask, kernel_erode, iterations=1)

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        cacat_bool = (img_gray < dark_threshold) & (inner_contour_mask > 0)
        cacat_mask = (cacat_bool.astype(np.uint8)) * 255
        total_cacat_pixels = int(np.sum(cacat_bool))

        luas_evaluasi = int(np.sum(inner_contour_mask > 0))
        persentase_cacat = float((total_cacat_pixels / luas_evaluasi) * 100.0) if luas_evaluasi > 0 else 0.0

    return {
        "luas_piksel": luas_piksel,
        "luas_evaluasi": luas_evaluasi,
        "total_cacat_pixels": total_cacat_pixels,
        "persentase_cacat": persentase_cacat,
        "cacat_mask": cacat_mask,
        "inner_contour_mask": inner_contour_mask,
        "main_contour": main_contour,
    }


def extract_pcv_features(
    img_bgr,
    crop_size: int = 300,
    save_crop_path=None,
    save_enhanced_path=None,
) -> dict:
    """Proses gambar yang baru ditangkap dari webcam menggunakan alur PCV lengkap (tanpa keputusan grade/prediksi).

    Tahapan:
      1. Segmentasi warna gabungan HSV + Morfologi Opening (5x5) & Closing (15x15) + Auto-Crop 1:1 (vision.segmentation).
      2. Fitur Warna HSV: persentase merah, oranye, kuning, dan hijau pada area tomat (vision.color).
      3. Fitur Tekstur: kanal Hijau (G) -> CLAHE -> Negatif -> Blackhat & Tophat + filter kilap
         -> kontur tomat (hijau) & kontur cacat (kuning) + rekonstruksi RGB via LAB (vision.texture).
      4. Hasil utama untuk dikirim ke ResNeXt: `img_enhanced_bgr` / `img_enhanced_rgb`.
    """
    from vision.color import extract_hsv_color_features
    from vision.segmentation import auto_crop_1to1_with_mask, segment_hsv_combined
    from vision.texture import extract_texture_and_enhance

    mask_clean, img_segmented_rgb, img_segmented_bgr = segment_hsv_combined(img_bgr)
    crop_asli_bgr, crop_asli_rgb, crop_seg_rgb, mask_crop_1to1 = auto_crop_1to1_with_mask(
        img_bgr, mask_clean, crop_size=crop_size
    )
    crop_seg_bgr = cv2.cvtColor(crop_seg_rgb, cv2.COLOR_RGB2BGR)

    color_feats = extract_hsv_color_features(crop_asli_bgr, mask_crop_1to1)
    texture_feats = extract_texture_and_enhance(crop_asli_bgr, mask_crop_1to1)

    if save_crop_path is not None:
        cv2.imwrite(str(save_crop_path), crop_seg_bgr)
    if save_enhanced_path is not None:
        cv2.imwrite(str(save_enhanced_path), texture_feats["img_enhanced_bgr"])

    return {
        "mask_clean": mask_clean,
        "mask_crop_1to1": mask_crop_1to1,
        "img_segmented_rgb": img_segmented_rgb,
        "img_segmented_bgr": img_segmented_bgr,
        "crop_asli_rgb": crop_asli_rgb,
        "crop_asli_bgr": crop_asli_bgr,
        "img_cropped_1to1_rgb": crop_seg_rgb,
        "img_cropped_1to1_bgr": crop_seg_bgr,
        **color_feats,
        **texture_feats,
    }
