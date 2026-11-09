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


# --- RENTANG WARNA HSV UNTUK MASKING GABUNGAN PCV ---
HSV_RED1_LOWER = np.array([0, 70, 50], dtype=np.uint8)
HSV_RED1_UPPER = np.array([10, 255, 255], dtype=np.uint8)
HSV_RED2_LOWER = np.array([170, 70, 50], dtype=np.uint8)
HSV_RED2_UPPER = np.array([180, 255, 255], dtype=np.uint8)

HSV_YELLOW_LOWER = np.array([10, 70, 50], dtype=np.uint8)
HSV_YELLOW_UPPER = np.array([35, 255, 255], dtype=np.uint8)

HSV_GREEN_LOWER = np.array([35, 40, 40], dtype=np.uint8)
HSV_GREEN_UPPER = np.array([85, 255, 255], dtype=np.uint8)


def segment_hsv_combined(image_bgr):
    """Masking gabungan HSV (Merah, Kuning/Oranye, dan Hijau) + pembersihan noise morfologi.

    Input:
        image_bgr: Citra BGR dari webcam.
    Output:
        tuple (mask_clean, img_segmented_rgb, img_segmented_bgr)
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Gambar input kosong atau tidak valid")

    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # 1. Rentang Warna Merah
    mask_red1 = cv2.inRange(img_hsv, HSV_RED1_LOWER, HSV_RED1_UPPER)
    mask_red2 = cv2.inRange(img_hsv, HSV_RED2_LOWER, HSV_RED2_UPPER)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # 2. Rentang Warna Kuning / Oranye
    mask_yellow = cv2.inRange(img_hsv, HSV_YELLOW_LOWER, HSV_YELLOW_UPPER)

    # 3. Rentang Warna Hijau
    mask_green = cv2.inRange(img_hsv, HSV_GREEN_LOWER, HSV_GREEN_UPPER)

    # Gabungkan semua masker warna
    mask = cv2.bitwise_or(mask_red, mask_yellow)
    mask = cv2.bitwise_or(mask, mask_green)

    # Pembersihan noise morfologi (Opening & Closing)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close)

    img_segmented_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_clean)
    img_segmented_bgr = cv2.bitwise_and(image_bgr, image_bgr, mask=mask_clean)
    return mask_clean, img_segmented_rgb, img_segmented_bgr


def auto_crop_1to1(img_segmented, mask_clean, crop_size: int = 300, scale_factor: float = 1.3):
    """Auto-crop 1:1 pas di tengah berdasarkan kontur terbesar, lalu resize ke (crop_size, crop_size).

    Input:
        img_segmented: Citra hasil segmentasi (RGB atau BGR).
        mask_clean: Mask biner hasil pembersihan morfologi (uint8).
        crop_size: Ukuran sisi hasil resize (default 300x300).
        scale_factor: Faktor perbesaran kotak terhadap ukuran kontur (default 1.3).
    Output:
        img_cropped_1to1: Citra hasil crop 1:1 berukuran (crop_size, crop_size, 3).
    """
    h_img, w_img = img_segmented.shape[:2]
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(main_contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(main_contour)
            center_x = x + w // 2
            center_y = y + h // 2

        _, _, w_box, h_box = cv2.boundingRect(main_contour)
        sisi = int(max(w_box, h_box) * scale_factor)
    else:
        center_x, center_y = w_img // 2, h_img // 2
        sisi = min(w_img, h_img) // 2

    sisi = max(1, sisi)
    x_new = center_x - (sisi // 2)
    y_new = center_y - (sisi // 2)

    x_new = max(0, min(x_new, w_img - sisi))
    y_new = max(0, min(y_new, h_img - sisi))
    sisi = max(1, min(sisi, w_img - x_new, h_img - y_new))

    img_cropped_tomat = img_segmented[y_new:y_new + sisi, x_new:x_new + sisi]
    img_cropped_1to1 = cv2.resize(img_cropped_tomat, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    return img_cropped_1to1
