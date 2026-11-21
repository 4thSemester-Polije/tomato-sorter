"""Gambar debug: kontur mask dan angka fitur di atas foto. Cara tercepat menemukan segmentasi yang salah."""
import cv2


def draw_debug(image, mask, features=None):
    out = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    if features:
        lines = [
            f"red {features['pct_red']:.0f}%",
            f"dark {features['pct_dark']:.1f}%",
            f"d {features['diameter_px']:.0f}px",
        ]
        for i, text in enumerate(lines):
            cv2.putText(out, text, (10, 25 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def draw_main_contour(image, main_contour, color=(0, 255, 0), thickness: int = 3):
    """Gambar kontur utama objek tomat saja (tanpa bounding box)."""
    out = image.copy()
    if main_contour is not None:
        cv2.drawContours(out, [main_contour], -1, color, thickness)
    return out


def draw_pcv_contours(image_bgr, cnts_tomat, cnts_cacat):
    """Gambar overlay kontur buah tomat (hijau) dan kontur cacat permukaan (kuning, BGR: (0, 255, 255))."""
    out = image_bgr.copy()
    if cnts_tomat:
        cv2.drawContours(out, cnts_tomat, -1, (0, 255, 0), 2)
    if cnts_cacat:
        cv2.drawContours(out, cnts_cacat, -1, (0, 255, 255), 2)
    return out
