"""Ekstraksi fitur. NAMA FITUR ADALAH KONTRAK ANTAR MODUL: jangan diubah tanpa memberi tahu tim."""

FEATURE_NAMES = (
    "pct_red",      # persen piksel merah di dalam area tomat (kematangan)
    "mean_h",       # rata-rata Hue
    "mean_s",       # rata-rata Saturation
    "mean_v",       # rata-rata Value
    "dark_ratio",   # proporsi piksel gelap/coklat di dalam area tomat (proksi cacat)
    "diameter_px",  # diameter dalam piksel
    "diameter_mm",  # diameter dalam mm (butuh calibration.mm_per_pixel)
)


def extract_features(image, mask) -> dict:
    """Input: gambar BGR dan mask dari segment(). Output: dict dengan kunci FEATURE_NAMES."""
    raise NotImplementedError("TODO: tim vision")
