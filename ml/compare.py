"""Bandingkan hasil ResNeXt-50 dan ResNeXt-101 dari file laporan di models/.

Perbandingan hanya sah bila kedua model memakai partisi data, hyperparameter,
ukuran input, set evaluasi primer, dan perangkat latensi yang sama. Fungsi ini
memeriksa semuanya dan menandai hasil "TIDAK SETARA" bila ada yang berbeda.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from ml.resnext import ARCHITECTURES, TASKS, artifact_paths, report_path

DEFAULT_PAIR = ("resnext50", "resnext101")
# Hyperparameter bawaan versi training lama (sebelum metadata mencatatnya).
LEGACY_HYPERPARAMETERS = {
    "seed": 42,
    "batch_size": 16,
    "head_epochs": 8,
    "fine_tune_epochs": 5,
    "head_learning_rate": 1e-3,
    "fine_tune_learning_rate": 1e-5,
    "fine_tune_layers": 30,
    "early_stopping_patience": 3,
    "input_size": [224, 224],
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"File laporan tidak ditemukan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _split_fingerprint(metadata: dict[str, Any]) -> dict[str, list[str]]:
    manifest = metadata["split_manifest"]
    return {
        part: sorted(entry["sha256"] for entry in manifest[part])
        for part in ("train", "validation", "test")
    }


def _hyperparameters(metadata: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    recorded = metadata.get("hyperparameters")
    if recorded is not None:
        return recorded, True
    legacy = dict(LEGACY_HYPERPARAMETERS)
    legacy["seed"] = metadata.get("seed", legacy["seed"])
    return legacy, False


def check_comparability(
    architectures: Sequence[str],
    reports: dict[str, dict[str, Any]],
    metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[str]:
    """Return a list of problems; an empty list means the comparison is fair."""
    problems: list[str] = []
    first = architectures[0]
    for other in architectures[1:]:
        for task in TASKS:
            if _split_fingerprint(metadata[first][task]) != _split_fingerprint(
                metadata[other][task]
            ):
                problems.append(
                    f"Partisi train/val/test tugas '{task}' berbeda antara {first} dan {other}."
                )
            base_params, _ = _hyperparameters(metadata[first][task])
            other_params, _ = _hyperparameters(metadata[other][task])
            if base_params != other_params:
                diff = sorted(
                    key
                    for key in set(base_params) | set(other_params)
                    if base_params.get(key) != other_params.get(key)
                )
                problems.append(
                    f"Hyperparameter tugas '{task}' berbeda ({', '.join(diff)}) "
                    f"antara {first} dan {other}."
                )
        base_primary = reports[first].get("primary_independent_test", {})
        other_primary = reports[other].get("primary_independent_test", {})
        if base_primary.get("independent_primary_test_images") != other_primary.get(
            "independent_primary_test_images"
        ):
            problems.append("Jumlah foto evaluasi primer berbeda.")
        devices = {
            reports[name].get("efficiency", {}).get("device") for name in (first, other)
        }
        if None in devices:
            problems.append(
                "Latensi/perangkat belum tercatat untuk salah satu model. "
                "Jalankan: python -m ml.evaluate --architecture <nama> di runtime yang sama."
            )
        elif len(devices) > 1:
            problems.append(
                f"Latensi diukur di perangkat berbeda ({', '.join(sorted(devices))}); "
                "latensi tidak sebanding."
            )
    return problems


def _row(architecture: str, report: dict[str, Any]) -> dict[str, Any]:
    secondary = report["secondary"]
    primary = report["primary_independent_test"]
    grade = primary["grade"]
    confusion = grade["confusion_matrix"]
    efficiency = report.get("efficiency", {})
    latency = efficiency.get("latency_ms", {})
    return {
        "architecture": architecture,
        "model": ARCHITECTURES[architecture],
        "secondary_ripeness_accuracy": secondary["ripeness"]["accuracy"],
        "secondary_ripeness_macro_f1": secondary["ripeness"]["macro_f1"],
        "secondary_defect_accuracy": secondary["defect"]["accuracy"],
        "secondary_defect_macro_f1": secondary["defect"]["macro_f1"],
        "primary_ripeness_macro_f1": primary["heads"]["ripeness"]["macro_f1"],
        "primary_defect_macro_f1": primary["heads"]["defect"]["macro_f1"],
        "primary_grade_accuracy": grade["accuracy"],
        "primary_grade_macro_f1": grade["macro_f1"],
        "primary_grade_recall_c": grade["recall_per_class"]["C"],
        "primary_grade_c_to_a_errors": confusion[2][0],
        "latency_median_ms": latency.get("median"),
        "latency_p95_ms": latency.get("p95"),
        "parameters_total": efficiency.get("parameters_total"),
        "device": efficiency.get("device"),
    }


def compare(
    models_dir: str | Path = "models",
    architectures: Sequence[str] = DEFAULT_PAIR,
) -> dict[str, Any]:
    if len(architectures) < 2:
        raise ValueError("Berikan minimal dua arsitektur untuk dibandingkan.")
    directory = Path(models_dir)
    reports = {name: _read_json(report_path(directory, name)) for name in architectures}
    metadata = {
        name: {
            task: _read_json(artifact_paths(directory, task, name)[1]) for task in TASKS
        }
        for name in architectures
    }
    problems = check_comparability(architectures, reports, metadata)
    notes = [
        f"Metadata {name}/{task} tidak mencatat hyperparameter (training versi lama); "
        "dipakai nilai bawaan lama untuk pemeriksaan."
        for name in architectures
        for task in TASKS
        if not _hyperparameters(metadata[name][task])[1]
    ]
    return {
        "comparable": not problems,
        "problems": problems,
        "notes": notes,
        "rows": [_row(name, reports[name]) for name in architectures],
    }


def _fmt(value: Any, spec: str = ".4f") -> str:
    return "-" if value is None else format(value, spec)


def render_markdown(result: dict[str, Any]) -> str:
    rows = result["rows"]
    header = "| Metrik | " + " | ".join(row["model"] for row in rows) + " |"
    divider = "|---|" + "---|" * len(rows)
    lines = [
        "# Perbandingan ResNeXt",
        "",
        "Status: **SETARA**" if result["comparable"] else "Status: **TIDAK SETARA**",
    ]
    for problem in result["problems"]:
        lines.append(f"- PERINGATAN: {problem}")
    for note in result["notes"]:
        lines.append(f"- Catatan: {note}")
    lines += ["", header, divider]
    metrics = (
        ("Sekunder - kematangan: akurasi", "secondary_ripeness_accuracy", ".4f"),
        ("Sekunder - kematangan: macro F1", "secondary_ripeness_macro_f1", ".4f"),
        ("Sekunder - cacat: akurasi", "secondary_defect_accuracy", ".4f"),
        ("Sekunder - cacat: macro F1", "secondary_defect_macro_f1", ".4f"),
        ("Primer - kematangan: macro F1", "primary_ripeness_macro_f1", ".4f"),
        ("Primer - cacat: macro F1", "primary_defect_macro_f1", ".4f"),
        ("Primer - grade A/B/C: akurasi", "primary_grade_accuracy", ".4f"),
        ("Primer - grade A/B/C: macro F1", "primary_grade_macro_f1", ".4f"),
        ("Primer - recall kelas C", "primary_grade_recall_c", ".4f"),
        ("Primer - salah prediksi C menjadi A", "primary_grade_c_to_a_errors", "d"),
        ("Latensi median per tomat (ms, 2 model)", "latency_median_ms", ".1f"),
        ("Latensi p95 per tomat (ms, 2 model)", "latency_p95_ms", ".1f"),
        ("Jumlah parameter (2 model)", "parameters_total", ",d"),
        ("Perangkat", "device", "s"),
    )
    for label, key, spec in metrics:
        cells = [_fmt(row[key], spec) for row in rows]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--architecture",
        nargs="+",
        choices=sorted(ARCHITECTURES),
        default=list(DEFAULT_PAIR),
    )
    parser.add_argument("--output", type=Path, help="Simpan tabel Markdown ke file ini.")
    args = parser.parse_args()
    result = compare(args.models_dir, args.architecture)
    markdown = render_markdown(result)
    print(markdown)
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
        args.output.with_suffix(".json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
