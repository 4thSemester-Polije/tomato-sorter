"""Aturan rubrik v1.1 (docs/RUBRIK.md). Grade = tingkat TERBURUK dari kematangan dan cacat.

Catatan evaluasi: fungsi ini mengubah tingkat menjadi grade untuk membuat label.
Jangan melatih model dengan label yang diturunkan dari fitur yang sama dengan masukan model
(hasilnya melingkar). Tingkat kematangan dan cacat harus dinilai manusia secara visual.
"""

_GRADE_BY_LEVEL = {1: "A", 2: "B", 3: "C"}


def grade_from_levels(ripeness_level: int, defect_level: int) -> str:
    for lv in (ripeness_level, defect_level):
        if lv not in _GRADE_BY_LEVEL:
            raise ValueError(f"Tingkat harus 1, 2, atau 3, bukan {lv!r}")
    return _GRADE_BY_LEVEL[max(ripeness_level, defect_level)]


def ripeness_level_from_red_pct(pct_red: float, ripe_min: float = 60.0, turning_min: float = 10.0) -> int:
    """Tingkat kematangan dari persen merah (acuan USDA). Batas bawah termasuk tingkat yang lebih baik."""
    if pct_red >= ripe_min:
        return 1
    if pct_red >= turning_min:
        return 2
    return 3
