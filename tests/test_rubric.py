import pytest

from ml.rubric import grade_from_levels, ripeness_level_from_red_pct


@pytest.mark.parametrize(
    "ripeness,defect,expected",
    [(1, 1, "A"), (1, 2, "B"), (2, 1, "B"), (2, 2, "B"), (1, 3, "C"), (3, 1, "C"), (3, 3, "C")],
)
def test_weakest_link(ripeness, defect, expected):
    assert grade_from_levels(ripeness, defect) == expected


@pytest.mark.parametrize("bad", [0, 4, -1])
def test_invalid_level(bad):
    with pytest.raises(ValueError):
        grade_from_levels(bad, 1)


@pytest.mark.parametrize("pct,expected", [(100, 1), (60, 1), (59.9, 2), (10, 2), (9.9, 3), (0, 3)])
def test_ripeness_boundaries(pct, expected):
    assert ripeness_level_from_red_pct(pct) == expected
