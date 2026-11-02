import json
from pathlib import Path

import numpy as np
import pytest

from ml.compare import compare, render_markdown
from ml.metrics import benchmark_latency, grade_metrics, summarize_latency
from ml.resnext import artifact_paths, display_name, report_path, validate_architecture


def _metrics(f1: float) -> dict:
    return {"accuracy": f1, "macro_f1": f1}


def _write_run(models_dir: Path, architecture: str, *, test_hashes=("a", "b"), device="CPU",
               batch_size=16, with_hyperparameters=True) -> None:
    manifest = {
        "train": [{"sha256": "t1"}],
        "validation": [{"sha256": "v1"}],
        "test": [{"sha256": h} for h in test_hashes],
    }
    for task in ("ripeness", "defect"):
        metadata = {"split_manifest": manifest, "seed": 42}
        if with_hyperparameters:
            metadata["hyperparameters"] = {
                "seed": 42, "batch_size": batch_size, "head_epochs": 8, "fine_tune_epochs": 5,
                "head_learning_rate": 1e-3, "fine_tune_learning_rate": 1e-5,
                "fine_tune_layers": 30, "early_stopping_patience": 3, "input_size": [224, 224],
            }
        artifact_paths(models_dir, task, architecture)[1].write_text(json.dumps(metadata))
    grade = grade_metrics(["A", "B", "C", "C"], ["A", "B", "C", "A"])
    report = {
        "secondary": {"ripeness": _metrics(0.9), "defect": _metrics(0.8)},
        "primary_independent_test": {
            "independent_primary_test_images": 4,
            "grade": grade,
            "heads": {"ripeness": _metrics(0.5), "defect": _metrics(0.7)},
        },
        "efficiency": {
            "parameters_total": 1000,
            "latency_ms": {"median": 120.0, "p95": 150.0},
            "device": device,
        },
    }
    report_path(models_dir, architecture).write_text(json.dumps(report))


def test_compare_marks_identical_setup_as_comparable(tmp_path):
    _write_run(tmp_path, "resnext50")
    _write_run(tmp_path, "resnext101")

    result = compare(tmp_path)

    assert result["comparable"] is True
    row = {item["architecture"]: item for item in result["rows"]}["resnext101"]
    assert row["primary_grade_recall_c"] == 0.5
    assert row["primary_grade_c_to_a_errors"] == 1
    assert "SETARA" in render_markdown(result)


def test_compare_flags_different_split(tmp_path):
    _write_run(tmp_path, "resnext50")
    _write_run(tmp_path, "resnext101", test_hashes=("a", "z"))

    result = compare(tmp_path)

    assert result["comparable"] is False
    assert any("Partisi" in problem for problem in result["problems"])
    assert "TIDAK SETARA" in render_markdown(result)


def test_compare_flags_different_hyperparameters_and_device(tmp_path):
    _write_run(tmp_path, "resnext50", batch_size=32, device="GPU: T4")
    _write_run(tmp_path, "resnext101")

    problems = compare(tmp_path)["problems"]

    assert any("batch_size" in problem for problem in problems)
    assert any("perangkat berbeda" in problem for problem in problems)


def test_compare_accepts_legacy_metadata_without_hyperparameters(tmp_path):
    _write_run(tmp_path, "resnext50")
    _write_run(tmp_path, "resnext101", with_hyperparameters=False)

    result = compare(tmp_path)

    assert result["comparable"] is True
    assert result["notes"]


def test_architecture_helpers():
    assert display_name("resnext50") == "ResNeXt50"
    assert artifact_paths("m", "ripeness", "resnext101")[0] == Path("m/ripeness_resnext101.keras")
    with pytest.raises(ValueError):
        validate_architecture("resnet18")
    with pytest.raises(ValueError):
        artifact_paths("m", "unknown", "resnext50")


def test_latency_summary_and_benchmark_with_fake_models():
    summary = summarize_latency([10, 20, 30, 40])
    assert summary["median"] == 25
    assert summary["runs"] == 4

    class FakeModel:
        calls = 0

        def predict(self, batch, verbose=0):
            FakeModel.calls += 1
            assert batch.shape == (1, 8, 8, 3)
            return np.zeros((1, 2))

    result = benchmark_latency([FakeModel(), FakeModel()], lambda x: x, (8, 8), runs=5, warmup=2)
    assert result["runs"] == 5
    assert FakeModel.calls == (5 + 2) * 2
