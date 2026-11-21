"""Ekstraksi fitur tekstur & deteksi cacat permukaan tomat + rekonstruksi citra RGB via LAB.

Alur tekstur PCV (tanpa keputusan grade/prediksi):
1. Ambil kanal Hijau (G) dari citra RGB crop 1:1 sebagai grayscale acuan tekstur.
2. Tingkatkan kontras lokal dengan CLAHE -> ubah ke citra negatif (gray_clahe_negatif).
3. Erosi mask tepi tomat (15x15) -> deteksi tekstur cacat menggunakan Blackhat + Tophat
   pada citra negatif CLAHE, sambil menyaring pantulan kilap cahaya lampu (persentil 12).
4. Temukan kontur tomat (hijau) & kontur cacat (kuning) serta hitung persentase area cacat.
5. Balikkan kembali negatif CLAHE ke positif lalu gabungkan detail teksturnya ke kanal L
   pada ruang warna LAB (hanya di dalam area tomat) agar kembali menjadi citra RGB alami
   yang teksturnya tajam dan siap dikirim ke model ResNeXt.
"""
from __future__ import annotations

import cv2
import numpy as np

DEFAULT_CLAHE_CLIP_LIMIT = 2.0
DEFAULT_CLAHE_TILE_GRID = (8, 8)
DEFAULT_ERODE_SIZE = 15
DEFAULT_DEFECT_KERNEL_SIZE = 15
DEFAULT_DEFECT_THRESHOLD = 32
DEFAULT_GLARE_PERCENTILE = 12.0
DEFAULT_LAB_DETAIL_WEIGHT = 0.4


def reconstruct_lab_enhanced(
    crop_asli_bgr,
    mask_crop_1to1,
    gray_clahe_negatif,
    *,
    clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
    tile_grid_size: tuple[int, int] = DEFAULT_CLAHE_TILE_GRID,
    detail_weight: float = DEFAULT_LAB_DETAIL_WEIGHT,
):
    """Kembalikan detail tekstur dari negatif CLAHE ke citra RGB alami melalui ruang warna LAB.

    Hanya kanal kecerahan (L) di dalam area tomat yang diperkuat (CLAHE + detail tekstur kanal hijau),
    sementara kanal warna (A, B) dan latar belakang dipertahankan asli agar ramah bagi ResNeXt.
    """
    tekstur_clahe_positif = 255 - gray_clahe_negatif
    blur_tekstur = cv2.GaussianBlur(tekstur_clahe_positif, (21, 21), 0)
    detail_tekstur = tekstur_clahe_positif.astype(np.float32) - blur_tekstur.astype(np.float32)

    lab = cv2.cvtColor(crop_asli_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    L_clahe = clahe.apply(L)
    L_boosted = np.clip(L_clahe.astype(np.float32) + detail_weight * detail_tekstur, 0, 255).astype(np.uint8)

    # Terapkan penguatan hanya di dalam area tomat agar latar belakang tetap bersih/mulus
    L_final = L.copy()
    inside = mask_crop_1to1 > 0
    if np.any(inside):
        L_final[inside] = L_boosted[inside]
    else:
        L_final = L_boosted

    img_enhanced_bgr = cv2.cvtColor(cv2.merge([L_final, A, B]), cv2.COLOR_LAB2BGR)
    img_enhanced_rgb = cv2.cvtColor(img_enhanced_bgr, cv2.COLOR_BGR2RGB)
    return img_enhanced_bgr, img_enhanced_rgb


def extract_texture_and_enhance(
    crop_asli_bgr,
    mask_crop_1to1,
    *,
    clahe_clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
    clahe_tile_grid: tuple[int, int] = DEFAULT_CLAHE_TILE_GRID,
    erode_size: int = DEFAULT_ERODE_SIZE,
    defect_kernel_size: int = DEFAULT_DEFECT_KERNEL_SIZE,
    ambang_cacat: int = DEFAULT_DEFECT_THRESHOLD,
    glare_percentile: float = DEFAULT_GLARE_PERCENTILE,
    lab_detail_weight: float = DEFAULT_LAB_DETAIL_WEIGHT,
) -> dict:
    """Jalankan pipeline tekstur PCV (Kanal Hijau -> CLAHE -> Negatif -> Blackhat/Tophat -> Kontur)
    dan rekonstruksi citra RGB enhanced via LAB (tanpa logika tingkat/grade).
    """
    crop_asli_rgb = cv2.cvtColor(crop_asli_bgr, cv2.COLOR_BGR2RGB)

    # 1. Kanal Hijau (G) sebagai grayscale tekstur -> CLAHE -> Negatif
    gray_hijau = crop_asli_bgr[:, :, 1].copy()
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid)
    gray_clahe = clahe.apply(gray_hijau)
    gray_clahe_negatif = 255 - gray_clahe

    # 2. Erosi mask tepi tomat & cari kontur buah tomat
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
    inner_mask = cv2.erode(mask_crop_1to1, kernel_erode, iterations=1)

    cnts_tomat, _ = cv2.findContours(mask_crop_1to1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(cnts_tomat, key=cv2.contourArea) if cnts_tomat else None
    luas_piksel = float(cv2.contourArea(main_contour)) if main_contour is not None else 0.0

    # 3. Ekstrak tekstur cacat dengan Blackhat & Tophat pada citra negatif CLAHE
    blur_negatif = cv2.GaussianBlur(gray_clahe_negatif, (5, 5), 0)
    kernel_cacat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (defect_kernel_size, defect_kernel_size))
    blackhat = cv2.morphologyEx(blur_negatif, cv2.MORPH_BLACKHAT, kernel_cacat)
    tophat = cv2.morphologyEx(blur_negatif, cv2.MORPH_TOPHAT, kernel_cacat)
    tekstur_cacat = cv2.add(blackhat, tophat)

    # 4. Abaikan pantulan cahaya lampu / kilap kulit tomat (pada negatif kilap menjadi gelap)
    cahaya_rata = cv2.GaussianBlur(gray_clahe_negatif, (55, 55), 0)
    inner_active = inner_mask > 0
    luas_dalam = int(np.sum(inner_active))

    if luas_dalam > 0:
        batas_gelap_kilap = float(np.percentile(gray_clahe_negatif[inner_active], glare_percentile))
        cacat_raw = (tekstur_cacat > ambang_cacat) & (cahaya_rata > batas_gelap_kilap) & inner_active
        cacat_mask = (cacat_raw.astype(np.uint8)) * 255
        luas_cacat = int(np.sum(cacat_mask > 0))
        persentase_cacat = float((luas_cacat / luas_dalam) * 100.0)
    else:
        batas_gelap_kilap = 0.0
        cacat_mask = np.zeros_like(mask_crop_1to1, dtype=np.uint8)
        luas_cacat = 0
        persentase_cacat = 0.0

    # 5. Overlay kontur tomat (hijau) dan kontur cacat (kuning)
    cnts_cacat, _ = cv2.findContours(cacat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_hasil_cacat_rgb = crop_asli_rgb.copy()
    cv2.drawContours(img_hasil_cacat_rgb, cnts_tomat, -1, (0, 255, 0), 2)
    cv2.drawContours(img_hasil_cacat_rgb, cnts_cacat, -1, (255, 255, 0), 2)
    img_hasil_cacat_bgr = cv2.cvtColor(img_hasil_cacat_rgb, cv2.COLOR_RGB2BGR)

    # 6. Rekonstruksi citra RGB alami + ketajaman tekstur via ruang warna LAB
    img_enhanced_bgr, img_enhanced_rgb = reconstruct_lab_enhanced(
        crop_asli_bgr,
        mask_crop_1to1,
        gray_clahe_negatif,
        clip_limit=clahe_clip_limit,
        tile_grid_size=clahe_tile_grid,
        detail_weight=lab_detail_weight,
    )

    return {
        "gray_hijau": gray_hijau,
        "gray_clahe": gray_clahe,
        "gray_clahe_negatif": gray_clahe_negatif,
        "tekstur_cacat": tekstur_cacat,
        "inner_mask": inner_mask,
        "cacat_mask": cacat_mask,
        "luas_piksel": luas_piksel,
        "luas_dalam": luas_dalam,
        "luas_cacat": luas_cacat,
        "persentase_cacat": persentase_cacat,
        "batas_gelap_kilap": batas_gelap_kilap,
        "main_contour": main_contour,
        "cnts_tomat": cnts_tomat,
        "cnts_cacat": cnts_cacat,
        "jumlah_kontur_cacat": len(cnts_cacat),
        "img_hasil_cacat_rgb": img_hasil_cacat_rgb,
        "img_hasil_cacat_bgr": img_hasil_cacat_bgr,
        "img_enhanced_bgr": img_enhanced_bgr,
        "img_enhanced_rgb": img_enhanced_rgb,
    }

