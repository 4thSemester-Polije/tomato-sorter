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
