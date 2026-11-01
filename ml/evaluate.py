"""Evaluate saved ResNeXt-101 models on the independent Primer photo set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.resnext import load_trusted_model
from ml.train import _evaluate_primary


def evaluate(data_root: str | Path, models_dir: str | Path = "models") -> dict[str, Any]:
    root = Path(data_root)
    directory = Path(models_dir)
    ripeness_model = load_trusted_model(directory / "ripeness_resnext101.keras")
    defect_model = load_trusted_model(directory / "defect_resnext101.keras")
    return _evaluate_primary(root, ripeness_model, defect_model, directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    report = evaluate(args.data_root, args.models_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
