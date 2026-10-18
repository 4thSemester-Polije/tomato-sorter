import cv2
import numpy as np
import pytest

from vision.features import FEATURE_NAMES, extract_features
from vision.segmentation import segment

BG = (235, 235, 235)       # latar putih pucat (BGR)
RED = (30, 30, 200)        # tomat merah (BGR)
GREEN = (60, 170, 70)      # tomat hijau (BGR)
BROWN = (20, 30, 50)       # bercak gelap (BGR)
CENTER, RADIUS = (320, 240), 90


def scene(color=RED, spot_radius=0):
    img = np.full((480, 640, 3), BG, dtype=np.uint8)
    cv2.circle(img, CENTER, RADIUS, color, -1)
    if spot_radius:
        cv2.circle(img, CENTER, spot_radius, BROWN, -1)
    return img


def truth_mask():
    m = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(m, CENTER, RADIUS, 255, -1)
    return m


def iou(a, b):
    a, b = a > 0, b > 0
    return (a & b).sum() / (a | b).sum()


def test_segment_by_saturation():
    assert iou(segment(scene()), truth_mask()) > 0.95


def test_segment_by_background():
    bg = np.full((480, 640, 3), BG, dtype=np.uint8)
    assert iou(segment(scene(), background=bg), truth_mask()) > 0.95


def test_empty_station_gives_empty_mask_and_error():
    bg = np.full((480, 640, 3), BG, dtype=np.uint8)
    mask = segment(bg.copy(), background=bg)
    assert mask.sum() == 0
    with pytest.raises(ValueError):
        extract_features(bg, mask)


def test_feature_keys_and_diameter():
    img = scene()
    f = extract_features(img, segment(img), mm_per_pixel=0.5)
    assert tuple(f.keys()) == FEATURE_NAMES
    assert f["diameter_px"] == pytest.approx(2 * RADIUS, rel=0.05)
    assert f["diameter_mm"] == pytest.approx(f["diameter_px"] * 0.5)


def test_diameter_mm_none_without_calibration():
    img = scene()
    assert extract_features(img, segment(img))["diameter_mm"] is None


def test_red_tomato_without_defect():
    img = scene()
    f = extract_features(img, segment(img))
    assert f["pct_red"] > 90
    assert f["pct_dark"] < 2


def test_green_tomato_is_not_red():
    img = scene(color=GREEN)
    assert extract_features(img, segment(img))["pct_red"] < 5


def test_dark_spot_is_counted_as_defect_not_red():
    img = scene(spot_radius=30)  # bercak ~11-12% luas tomat
    f = extract_features(img, segment(img))
    assert 9 < f["pct_dark"] < 16
    assert 78 < f["pct_red"] < 92


def test_mean_hue_is_circular_for_red():
    hsv = np.zeros((100, 100, 3), dtype=np.uint8)
    hsv[:, :50] = (2, 200, 200)     # merah sisi bawah lingkaran hue
    hsv[:, 50:] = (177, 200, 200)   # merah sisi atas lingkaran hue
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    f = extract_features(img, np.full((100, 100), 255, dtype=np.uint8))
    h = f["mean_h"]
    assert min(h, 180 - h) < 3      # dekat 0/179, BUKAN ~90 seperti rata-rata biasa
