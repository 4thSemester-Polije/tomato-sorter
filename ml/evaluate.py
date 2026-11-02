"""Evaluate saved ResNeXt models on the independent Primer photo set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.resnext import (
    ARCHITECTURES,
    DEFAULT_ARCHITECTURE,
    artifact_paths,
    load_trusted_model,
    report_path,
    validate_architecture,
)
from ml.train import _evaluate_primary, measure_efficiency


def evaluate(
    data_root: str | Path,
    models_dir: str | Path = "models",
    architecture: str = DEFAULT_ARCHITECTURE,
) -> dict[str, Any]:
    validate_architecture(architecture)
    root = Path(data_root)
    directory = Path(models_dir)
    ripeness_model = load_trusted_model(artifact_paths(directory, "ripeness", architecture)[0])
    defect_model = load_trusted_model(artifact_paths(directory, "defect", architecture)[0])
    return {
        "primary_independent_test": _evaluate_primary(
            root, ripeness_model, defect_model, directory, architecture
        ),
        "efficiency": measure_efficiency(ripeness_model, defect_model, architecture),
    }


def update_report(
    models_dir: str | Path, architecture: str, fresh: dict[str, Any]
) -> Path:
    """Merge fresh primary/efficiency results into evaluation_<arch>.json, keeping 'secondary'."""
    path = report_path(models_dir, architecture)
    report: dict[str, Any] = {}
    if path.is_file():
        report = json.loads(path.read_text(encoding="utf-8"))
    report.update(fresh)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--architecture",
        choices=sorted(ARCHITECTURES),
        default=DEFAULT_ARCHITECTURE,
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Hanya cetak hasil; jangan memperbarui evaluation_<arsitektur>.json.",
    )
    args = parser.parse_args()
    fresh = evaluate(args.data_root, args.models_dir, args.architecture)
    if not args.no_write:
        path = update_report(args.models_dir, args.architecture, fresh)
        print(f"# Laporan diperbarui: {path}")
    print(json.dumps(fresh, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
