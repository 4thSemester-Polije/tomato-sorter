import cv2
import numpy as np

from vision.crop import (
    NEUTRAL_BACKGROUND_BGR,
    crop_largest_tomato,
    crop_region,
    find_tomato_regions,
    is_clean_single_tomato,
)

RED = (30, 40, 200)  # BGR, jenuh


def _canvas(width=400, height=300):
    return np.full((height, width, 3), 255, dtype=np.uint8)


def _tomato(image, center, axes=(60, 66)):
    cv2.ellipse(image, center, axes, 0, 0, 360, RED, -1)
    return image


def test_single_tomato_is_found_and_clean():
    image = _tomato(_canvas(), (200, 150))

    regions = find_tomato_regions(image)

    assert len(regions) == 1
    assert regions[0].solidity > 0.95
    assert not regions[0].touches_border
    assert is_clean_single_tomato(regions[0])


def test_touching_tomatoes_are_rejected_as_not_single():
    image = _canvas(500, 300)
    _tomato(image, (190, 150), (50, 50))
    _tomato(image, (290, 150), (50, 50))

    regions = find_tomato_regions(image)

    assert len(regions) == 1  # menyatu, sengaja tidak dipisahkan
    assert regions[0].solidity < 0.92
    assert not is_clean_single_tomato(regions[0])


def test_separate_tomatoes_are_two_regions():
    image = _canvas(600, 300)
    _tomato(image, (150, 150), (50, 50))
    _tomato(image, (450, 150), (50, 50))

    regions = find_tomato_regions(image)

    assert len(regions) == 2
    assert all(is_clean_single_tomato(region) for region in regions)


def test_tomato_touching_the_image_border_is_not_clean():
    image = _tomato(_canvas(), (20, 150))

    regions = find_tomato_regions(image)

    assert regions and regions[0].touches_border
    assert not is_clean_single_tomato(regions[0])


def test_blank_image_has_no_tomato():
    assert find_tomato_regions(_canvas()) == []
    assert crop_largest_tomato(_canvas()) is None


def test_crop_is_square_and_keeps_the_shape_undistorted():
    image = _tomato(_canvas(), (200, 150), axes=(40, 80))  # lonjong tegak
    region = find_tomato_regions(image)[0]

    crop = crop_region(image, region, size=200, neutral_background=False)

    assert crop.shape == (200, 200, 3)
    red = (np.abs(crop.astype(int) - np.array(RED)).sum(axis=2) < 60).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(red)
    assert abs((w / h) - (40 / 80)) < 0.08


def test_neutral_background_replaces_everything_outside_the_tomato():
    image = _tomato(_canvas(), (200, 150), axes=(40, 40))
    region = find_tomato_regions(image)[0]

    neutral = crop_region(image, region, size=128, neutral_background=True)
    kept = crop_region(image, region, size=128, neutral_background=False)

    assert tuple(neutral[2, 2]) == NEUTRAL_BACKGROUND_BGR
    assert tuple(kept[2, 2]) == (255, 255, 255)
    assert neutral[64, 64].tolist() == list(RED)  # isi tomat tidak berubah


def test_crop_largest_tomato_returns_requested_size():
    image = _tomato(_canvas(), (200, 150))

    crop = crop_largest_tomato(image, size=96)

    assert crop is not None and crop.shape == (96, 96, 3)


def test_inference_can_relax_the_solidity_check_but_never_the_border_check():
    merged = _canvas(500, 300)
    _tomato(merged, (190, 150), (50, 50))
    _tomato(merged, (290, 150), (50, 50))
    at_border = _tomato(_canvas(), (20, 150))

    assert crop_largest_tomato(merged) is None  # ketat: untuk membangun dataset
    relaxed = crop_largest_tomato(merged, size=64, min_solidity=0.0)  # longgar: untuk alat
    assert relaxed is not None and relaxed.shape == (64, 64, 3)
    assert crop_largest_tomato(at_border, min_solidity=0.0) is None
