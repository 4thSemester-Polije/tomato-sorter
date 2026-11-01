from pathlib import Path

import pytest

from ml.train import Sample, _collect_samples, _split_samples


def test_collect_samples_deduplicates_identical_files(tmp_path: Path):
    class_a = tmp_path / "class-a"
    class_b = tmp_path / "class-b"
    class_a.mkdir()
    class_b.mkdir()
    (class_a / "first.jpg").write_bytes(b"same-image")
    (class_a / "duplicate.jpg").write_bytes(b"same-image")
    (class_b / "other.jpg").write_bytes(b"different-image")

    samples = _collect_samples({"a": class_a, "b": class_b})

    assert len(samples) == 2
    assert {sample.label for sample in samples} == {"a", "b"}


def test_collect_samples_excludes_conflicting_duplicate_labels(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    class_a = tmp_path / "class-a"
    class_b = tmp_path / "class-b"
    class_a.mkdir()
    class_b.mkdir()
    (class_a / "same-a.jpg").write_bytes(b"same-image")
    (class_b / "same-b.jpg").write_bytes(b"same-image")
    (class_a / "unique-a.jpg").write_bytes(b"unique-a")
    (class_b / "unique-b.jpg").write_bytes(b"unique-b")

    samples = _collect_samples({"a": class_a, "b": class_b})

    assert {sample.path.name for sample in samples} == {"unique-a.jpg", "unique-b.jpg"}
    assert "conflicting labels" in caplog.text


def test_split_samples_is_stratified_reproducible_and_disjoint():
    samples = [
        Sample(Path(f"{label}-{index}.jpg"), label, f"{label}-{index}")
        for label in ("ripe", "unripe")
        for index in range(20)
    ]

    first = _split_samples(samples, seed=13)
    second = _split_samples(samples, seed=13)

    assert [len(part) for part in first] == [28, 6, 6]
    assert [[sample.path for sample in part] for part in first] == [
        [sample.path for sample in part] for part in second
    ]
    assert not (set(first[0]) & set(first[1]))
    assert not (set(first[0]) & set(first[2]))
    assert not (set(first[1]) & set(first[2]))
    for partition in first:
        assert {sample.label for sample in partition} == {"ripe", "unripe"}
