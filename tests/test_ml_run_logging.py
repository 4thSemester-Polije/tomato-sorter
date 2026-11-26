import csv
import json
import logging
from pathlib import Path

import pytest

from ml.metrics import binary_ranking_metrics, classification_metrics
from ml.resnext import artifact_paths
from ml.run_logging import (
    HISTORY_COLUMNS,
    EpochRecorder,
    class_metrics_table,
    confusion_table,
    history_table,
    read_history,
    render_table,
    start_run,
)
from ml.train import (
    Sample,
    _defect_primary_sections,
    _split_samples_from_manifest,
    train,
)


def test_render_table_is_markdown_with_numeric_columns_right_aligned():
    table = render_table(["kelas", "recall"], [["healthy", 0.5], ["reject", 1.0]])
    lines = table.splitlines()

    assert lines[0].startswith("| kelas")
    assert lines[1].replace("-", "").replace("|", "").replace(":", "").strip() == ""
    assert lines[1].split("|")[2].endswith(":")  # kolom angka rata kanan
    assert "0.5000" in lines[2] and "1.0000" in lines[3]
    assert len({len(line) for line in lines}) == 1  # semua baris sama lebar


def test_render_table_handles_none_and_escapes_pipes():
    table = render_table(["a", "b"], [["x|y", None]])

    row = table.splitlines()[2]
    assert "x\\|y" in row
    assert row.split("|")[-2].strip() == "-"


def test_render_table_rejects_ragged_rows():
    with pytest.raises(ValueError):
        render_table(["a", "b"], [["only-one"]])


def test_metric_tables_come_from_classification_metrics():
    metrics = classification_metrics([0, 0, 1, 1, 1], [0, 1, 1, 1, 0], ["healthy", "reject"])

    per_class = class_metrics_table(metrics)
    confusion = confusion_table(metrics)

    assert "healthy" in per_class and "macro avg" in per_class and "accuracy" in per_class
    assert "support" in per_class
    assert "sebenarnya" in confusion and "recall" in confusion
    assert "| healthy" in confusion


def test_epoch_recorder_writes_header_once_and_appends_rows(tmp_path: Path):
    path = tmp_path / "history.csv"
    head = EpochRecorder(path, "defect", "head")
    head.record(1, {"loss": 0.9, "accuracy": 0.5, "val_loss": 0.8, "val_accuracy": 0.6}, 1e-3, 12.5)
    head.record(2, {"loss": 0.7, "accuracy": 0.7, "val_loss": 0.75, "val_accuracy": 0.65}, 1e-3, 11.0)
    fine = EpochRecorder(path, "defect", "fine_tune", epoch_offset=2)
    fine.record(1, {"loss": 0.6, "accuracy": 0.8}, 1e-5, None)

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    assert rows[0] == list(HISTORY_COLUMNS)
    assert len(rows) == 4
    history = read_history(path)
    assert [row["epoch"] for row in history] == [1, 2, 3]
    assert history[2]["phase"] == "fine_tune" and history[2]["val_loss"] is None
    assert history[0]["val_accuracy"] == pytest.approx(0.6)
    assert "fine_tune" in history_table(history)


def test_epoch_recorder_logs_a_readable_line(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.INFO, logger="ml.run_logging"):
        EpochRecorder(tmp_path / "h.csv", "defect", "head").record(
            1, {"loss": 0.5, "accuracy": 0.8, "val_loss": 0.4, "val_accuracy": 0.85}, 1e-3, 3.0
        )

    assert "[defect/head] epoch 1" in caplog.text
    assert "val_acc 0.8500" in caplog.text


def test_start_run_writes_log_and_config_then_detaches(tmp_path: Path):
    root = logging.getLogger()
    handlers_before = list(root.handlers)

    run = start_run(tmp_path, "uji", "resnext101", {"seed": 7}, console=False)
    logging.getLogger("ml.train").info("halo dari training")
    run.close()

    assert run.directory.parent == tmp_path
    assert run.directory.name.startswith("uji_resnext101_")
    assert "halo dari training" in run.log_path.read_text(encoding="utf-8")
    config = json.loads((run.directory / "config.json").read_text(encoding="utf-8"))
    assert config["seed"] == 7
    assert "python" in config["environment"] and "git" in config["environment"]
    assert root.handlers == handlers_before


def test_start_run_never_reuses_an_existing_folder(tmp_path: Path, monkeypatch):
    class FixedClock:
        @staticmethod
        def now():
            from datetime import datetime

            return datetime(2026, 11, 24, 15, 30, 0)

    monkeypatch.setattr("ml.run_logging.datetime", FixedClock)
    first = start_run(tmp_path, "r", "resnext101", {}, console=False)
    second = start_run(tmp_path, "r", "resnext101", {}, console=False)
    first.close()
    second.close()

    assert first.directory != second.directory


def _samples():
    return [
        Sample(Path(f"/x/{label}/{label}_{index}.jpg"), label, f"{label}{index}")
        for label in ("healthy", "reject")
        for index in range(6)
    ]


def _write_manifest(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["file", "split", "group"])
        writer.writerows(rows)
    return path


def _manifest_rows(samples, group_of=lambda sample, index: f"g{index}"):
    splits = ["train", "train", "train", "train", "validation", "test"]
    rows = []
    for label in ("healthy", "reject"):
        for index, split in enumerate(splits):
            rows.append((f"{label}/{label}_{index}.jpg", split, f"{label}-{index}"))
    return rows


def test_manifest_split_assigns_by_file_and_keeps_groups_together(tmp_path: Path):
    manifest = _write_manifest(tmp_path / "m.csv", _manifest_rows(_samples()))

    train_part, validation_part, test_part = _split_samples_from_manifest(_samples(), manifest)

    assert (len(train_part), len(validation_part), len(test_part)) == (8, 2, 2)
    assert {sample.label for sample in test_part} == {"healthy", "reject"}


def test_manifest_split_rejects_groups_spanning_splits(tmp_path: Path):
    rows = _manifest_rows(_samples())
    rows[0] = (rows[0][0], "train", "shared")
    rows[5] = (rows[5][0], "test", "shared")
    manifest = _write_manifest(tmp_path / "m.csv", rows)

    with pytest.raises(ValueError, match="leakage"):
        _split_samples_from_manifest(_samples(), manifest)


def test_manifest_split_rejects_images_missing_from_manifest(tmp_path: Path):
    manifest = _write_manifest(tmp_path / "m.csv", _manifest_rows(_samples())[:-1])

    with pytest.raises(ValueError, match="not listed"):
        _split_samples_from_manifest(_samples(), manifest)


def test_manifest_split_rejects_unknown_split_and_missing_columns(tmp_path: Path):
    bad_split = _write_manifest(tmp_path / "a.csv", [("healthy/healthy_0.jpg", "val", "g")])
    with pytest.raises(ValueError, match="Unknown split"):
        _split_samples_from_manifest(_samples(), bad_split)

    no_group = tmp_path / "b.csv"
    no_group.write_text("file,split\nhealthy/healthy_0.jpg,train\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks column"):
        _split_samples_from_manifest(_samples(), no_group)


def test_manifest_split_requires_both_classes_in_every_split(tmp_path: Path):
    rows = [(r[0], r[1], r[2]) for r in _manifest_rows(_samples())]
    rows = [(f, "train" if f.startswith("reject") and s == "test" else s, g) for f, s, g in rows]
    manifest = _write_manifest(tmp_path / "m.csv", rows)

    with pytest.raises(ValueError, match="both classes"):
        _split_samples_from_manifest(_samples(), manifest)


def test_binary_ranking_metrics_separable_scores():
    result = binary_ranking_metrics(
        [0, 0, 0, 0, 1, 1, 1], [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9], max_fpr=0.10
    )

    assert result["auc"] == pytest.approx(1.0)
    assert result["constrained"]["recall"] == pytest.approx(1.0)
    assert result["constrained"]["false_positive_rate"] == 0.0
    assert result["fixed_threshold"]["recall"] == pytest.approx(1.0)


def test_binary_ranking_metrics_respects_the_false_positive_budget():
    expected = [0] * 10 + [1] * 4
    scores = [0.9, 0.85, 0.1, 0.1, 0.2, 0.3, 0.1, 0.2, 0.1, 0.3, 0.95, 0.8, 0.5, 0.4]

    result = binary_ranking_metrics(expected, scores, max_fpr=0.10)

    # Dua negatif berskor tinggi: ambang harus di atas 0,85 agar FPR <= 10%.
    assert result["constrained"]["false_positive_rate"] <= 0.10
    assert result["constrained"]["recall"] == pytest.approx(0.25)
    assert result["constrained"]["threshold"] > 0.85


def test_binary_ranking_metrics_without_both_classes_has_no_auc():
    result = binary_ranking_metrics([0, 0, 0], [0.1, 0.2, 0.3])

    assert result["auc"] is None and result["constrained"] is None
    assert result["fixed_threshold"]["recall"] is None


def test_train_refuses_to_overwrite_existing_model_when_training_one_task(tmp_path: Path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    artifact_paths(models_dir, "defect", "resnext101")[0].write_bytes(b"old-model")

    with pytest.raises(FileExistsError, match="--overwrite-models"):
        train(tmp_path, models_dir, tasks=["defect"])
    assert artifact_paths(models_dir, "defect", "resnext101")[0].read_bytes() == b"old-model"


def test_train_validates_tasks_and_manifest_usage(tmp_path: Path):
    with pytest.raises(ValueError, match="tasks"):
        train(tmp_path, tmp_path, tasks=["grade"])
    with pytest.raises(ValueError, match="single task"):
        train(tmp_path, tmp_path, tasks=["ripeness", "defect"], split_manifest=tmp_path / "m.csv")


def _primary(recall: float, positives: int = 9):
    return {
        "per_level": {
            1: {"images": 73, "predicted_reject": 10, "predicted_reject_rate": 0.137,
                "mean_p_reject": 0.2, "median_p_reject": 0.1},
            2: {"images": 26, "predicted_reject": 9, "predicted_reject_rate": 0.346,
                "mean_p_reject": 0.4, "median_p_reject": 0.3},
            3: {"images": 9, "predicted_reject": 7, "predicted_reject_rate": 0.778,
                "mean_p_reject": 0.8, "median_p_reject": 0.9},
        },
        "level1_vs_level3": {
            "positives": positives, "negatives": 73, "max_fpr": 0.10, "auc": 0.9,
            "fixed_threshold": {"recall": 0.78, "false_positive_rate": 0.14},
            "constrained": {"recall": recall, "threshold": 0.83},
        },
        "level1_vs_level2_or_3": {"auc": 0.8},
        "gate": {
            "min_recall": 0.70,
            "max_fpr": 0.10,
            "min_positives": 5,
            "enough_positives": positives >= 5,
            "passed": positives >= 5 and recall >= 0.70,
        },
    }


def test_defect_primary_sections_report_gate_verdict():
    passed = dict(_defect_primary_sections(_primary(0.78)))
    failed = dict(_defect_primary_sections(_primary(0.33)))

    assert "**LULUS**" in passed["Primer - gate go/no-go"]
    assert "**BELUM LULUS**" in failed["Primer - gate go/no-go"]
    assert "level 3" in passed["Primer - model cacat per level cacat sebenarnya"]
    assert "AUC level 1 vs 3" in passed["Primer - ringkasan model cacat"]


def test_defect_primary_gate_is_not_decided_with_too_few_positives():
    sections = dict(_defect_primary_sections(_primary(1.0, positives=2)))

    assert "TIDAK DAPAT DINILAI" in sections["Primer - gate go/no-go"]
    assert "**LULUS**" not in sections["Primer - gate go/no-go"]
