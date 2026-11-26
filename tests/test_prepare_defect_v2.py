import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from ml.prepare_defect_v2 import (
    assign_groups,
    prepare,
    source_index,
)
from ml.train import Sample, _split_samples_from_manifest
from vision.crop import NEUTRAL_BACKGROUND_BGR

RED = (30, 40, 200)


def _write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    path.write_bytes(encoded.tobytes())


def _scene(centers, radius=50, seed=0, spot=False):
    """Tomat sintetis dengan bercak acak di dalamnya, supaya tiap gambar punya "tekstur" sendiri."""
    rng = np.random.default_rng(seed)
    image = np.full((300, 500, 3), 255, dtype=np.uint8)
    for number, (cx, cy) in enumerate(centers):
        shade = int(rng.integers(0, 40))
        cv2.ellipse(image, (cx, cy), (radius, radius + 6), 0, 0, 360, (30 + shade, 40, 200 - shade), -1)
        for _ in range(4):
            dx, dy = int(rng.integers(-radius // 2, radius // 2)), int(rng.integers(-radius // 2, radius // 2))
            tone = int(rng.integers(60, 255))
            cv2.circle(image, (cx + dx, cy + dy), int(rng.integers(5, 14)), (tone // 3, tone // 2, tone), -1)
        if spot and number == 0:
            cv2.circle(image, (cx, cy), 12, (10, 10, 10), -1)
    return image


def _agml(root: Path, fresh=8, rotten=8, extras=True) -> Path:
    for index in range(fresh):
        _write(root / "Fresh" / f"AGML_{636 + index * 50:04d}.jpg", _scene([(180 + 10 * index, 150)], seed=index))
    for index in range(rotten):
        _write(root / "Rotten" / f"AGML_{index * 50:04d}.jpg",
               _scene([(200, 130 + 5 * index)], seed=100 + index, spot=True))
    if extras:
        _write(root / "Rotten" / "AGML_0401.jpg", _scene([(120, 150), (380, 150)]))  # dua terpisah
        _write(root / "Rotten" / "AGML_0451.jpg", _scene([(200, 150), (300, 150)]))  # menyatu
        _write(root / "Fresh" / "AGML_1999.jpg", np.full((300, 500, 3), 255, dtype=np.uint8))  # kosong
    return root


def _kwargs(**overrides):
    base = dict(group_block=0, dup_threshold=0.0, qa_samples=4)
    base.update(overrides)
    return base


def _manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_source_index_reads_trailing_number():
    assert source_index("AGML_0003") == 3
    assert source_index("foto") is None


def test_prepare_builds_dataset_manifest_and_report(tmp_path: Path):
    agml = _agml(tmp_path / "agml")
    out = tmp_path / "v2"

    summary = prepare(agml, out, size=128, **_kwargs())

    assert summary["crops"] == {"healthy": 8, "reject": 8}
    assert summary["excluded"]["multi_tomat_rotten"] == 2  # terpisah + menyatu
    assert summary["excluded"]["tidak_ada_tomat"] == 1
    assert len(list((out / "healthy").glob("*.jpg"))) == 8
    assert (out / "prepare_report.md").read_text(encoding="utf-8").startswith("# Laporan")
    assert (out / "qa" / "crops_reject.jpg").is_file()
    assert (out / "qa" / "excluded_multi_tomat_rotten.jpg").is_file()

    rows = _manifest(out / "split_manifest.csv")
    assert {row["split"] for row in rows} == {"train", "validation", "test"}
    samples = [
        Sample(out / row["file"], row["file"].split("/")[0], row["file"]) for row in rows
    ]
    train_part, validation_part, test_part = _split_samples_from_manifest(samples, out / "split_manifest.csv")
    assert train_part and validation_part and test_part


def test_multi_rotten_include_keeps_every_clean_tomato(tmp_path: Path):
    agml = _agml(tmp_path / "agml")

    summary = prepare(agml, tmp_path / "v2", size=96, multi_rotten="include", **_kwargs())

    assert summary["crops"]["reject"] == 8 + 2  # dua tomat terpisah ikut; yang menyatu tetap ditolak
    assert summary["excluded"]["tomat_menyatu_atau_terpotong"] == 1


def test_neutral_and_keep_background_differ_at_the_corner(tmp_path: Path):
    agml = _agml(tmp_path / "agml", extras=False)

    prepare(agml, tmp_path / "neutral", size=128, neutral_background=True, **_kwargs())
    prepare(agml, tmp_path / "keep", size=128, neutral_background=False, **_kwargs())

    name = sorted(p.name for p in (tmp_path / "neutral" / "healthy").glob("*.jpg"))[0]
    neutral = cv2.imread(str(tmp_path / "neutral" / "healthy" / name))
    kept = cv2.imread(str(tmp_path / "keep" / "healthy" / name))
    assert np.abs(neutral[2, 2].astype(int) - np.array(NEUTRAL_BACKGROUND_BGR)).max() <= 6
    assert kept[2, 2].min() >= 245


def test_duplicate_copies_share_a_group_and_a_split(tmp_path: Path):
    agml = _agml(tmp_path / "agml", extras=False)
    source = next((agml / "Fresh").glob("*.jpg"))
    shutil.copy(source, agml / "Fresh" / "AGML_1500.jpg")

    prepare(agml, tmp_path / "v2", size=128, **_kwargs(dup_threshold=0.985))

    rows = {row["file"]: row for row in _manifest(tmp_path / "v2" / "split_manifest.csv")}
    original = rows[f"healthy/{source.stem}_t1.jpg"]
    copy = rows["healthy/AGML_1500_t1.jpg"]
    assert original["group"] == copy["group"]
    assert original["split"] == copy["split"]


def test_same_image_in_both_classes_is_removed_as_label_conflict(tmp_path: Path):
    agml = _agml(tmp_path / "agml", extras=False)
    shutil.copy(next((agml / "Fresh").glob("*.jpg")), agml / "Rotten" / "AGML_0777.jpg")

    summary = prepare(agml, tmp_path / "v2", size=128, **_kwargs(dup_threshold=0.985))

    assert summary["excluded"]["label_konflik"] == 2
    assert not (tmp_path / "v2" / "reject" / "AGML_0777_t1.jpg").exists()
    assert len(list((tmp_path / "v2" / "healthy").glob("*.jpg"))) == 7


def test_nearby_numbered_photos_form_one_session_group(tmp_path: Path):
    agml = tmp_path / "agml"
    for index in (0, 1, 2, 300, 600, 900, 1200):
        _write(agml / "Rotten" / f"AGML_{index:04d}.jpg", _scene([(200, 150)], seed=index, spot=True))
    for index in range(1000, 1007):
        _write(agml / "Fresh" / f"AGML_{index * 50:04d}.jpg", _scene([(200, 150)], seed=index))

    prepare(agml, tmp_path / "v2", size=96, **_kwargs(group_block=10))

    rows = {row["file"]: row["group"] for row in _manifest(tmp_path / "v2" / "split_manifest.csv")}
    assert rows["reject/AGML_0000_t1.jpg"] == rows["reject/AGML_0001_t1.jpg"] == rows["reject/AGML_0002_t1.jpg"]
    assert rows["reject/AGML_0300_t1.jpg"] != rows["reject/AGML_0000_t1.jpg"]


def test_refuses_to_overwrite_and_is_deterministic(tmp_path: Path):
    agml = _agml(tmp_path / "agml")
    out = tmp_path / "v2"
    prepare(agml, out, size=96, **_kwargs())
    first = (out / "split_manifest.csv").read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        prepare(agml, out, size=96, **_kwargs())
    prepare(agml, out, size=96, overwrite=True, **_kwargs())

    assert (out / "split_manifest.csv").read_text(encoding="utf-8") == first
    assert "\r" not in first


def test_too_few_groups_per_class_is_a_clear_error(tmp_path: Path):
    agml = _agml(tmp_path / "agml", fresh=8, rotten=2, extras=False)

    with pytest.raises(ValueError, match="minimal 3"):
        prepare(agml, tmp_path / "v2", size=96, **_kwargs())


def test_missing_class_folder_is_reported(tmp_path: Path):
    (tmp_path / "agml" / "Fresh").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="rotten"):
        prepare(tmp_path / "agml", tmp_path / "v2", **_kwargs())


def test_assign_groups_unions_by_source_even_across_boxes():
    from ml.prepare_defect_v2 import CropRecord

    thumb = np.zeros(4, dtype=np.float32)
    records = [
        CropRecord("reject/a_t1.jpg", "reject", "a", 1, (0, 0, 1, 1), 0.1, 0.99, 2, thumb),
        CropRecord("reject/a_t2.jpg", "reject", "a", 1, (0, 0, 1, 1), 0.1, 0.99, 2, thumb),
        CropRecord("reject/b_t1.jpg", "reject", "b", 900, (0, 0, 1, 1), 0.1, 0.99, 1, thumb),
    ]

    groups = assign_groups(records, group_block=20, dup_threshold=0.0)

    assert groups[0] == groups[1] != groups[2]
