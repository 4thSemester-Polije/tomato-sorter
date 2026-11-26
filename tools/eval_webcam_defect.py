"""Evaluasi uji webcam model cacat dari results.csv hasil `tools.camera_predict --log-dir`.

Isi dulu kolom true_defect_level (1 = tanpa cacat, 2 = cacat ringan, 3 = cacat jelas) di results.csv.
Ambang DIBEKUKAN sebelum melihat data (bawaan 0,5, sama dengan API) supaya hasilnya jujur; tabel
sapuan ambang hanya untuk informasi dan tidak boleh dipakai memilih ambang pada data yang sama.

    python -m tools.eval_webcam_defect --csv data/webcam_test/results.csv [--threshold 0.5] [--out laporan.md]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from ml.metrics import binary_ranking_metrics, wilson_interval
from ml.run_logging import render_table

GATE_MIN_RECALL = 0.70
GATE_MAX_FPR = 0.10
GATE_MIN_POSITIVES = 10
SWEEP = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def load_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _level(row: dict[str, str]) -> int | None:
    text = (row.get("true_defect_level") or "").strip()
    return int(text) if text in {"1", "2", "3"} else None


def _score(row: dict[str, str]) -> float | None:
    text = (row.get("defect_p_reject") or "").strip()
    return float(text) if text else None


def _proportion(successes: int, total: int) -> str:
    interval = wilson_interval(successes, total)
    if interval is None:
        return "-"
    return f"{successes}/{total} = {successes / total:.0%} (95%: {interval[0]:.0%}-{interval[1]:.0%})"


def evaluate(rows: list[dict[str, str]], threshold: float = 0.5) -> dict[str, Any]:
    labeled = [(row, _level(row), _score(row)) for row in rows if _level(row) is not None]
    per_level: dict[int, dict[str, int]] = {}
    for level in (1, 2, 3):
        group = [(score) for _, lvl, score in labeled if lvl == level]
        per_level[level] = {
            "n": len(group),
            "not_assessed": sum(1 for score in group if score is None),
            "flagged": sum(1 for score in group if score is not None and score >= threshold),
        }
    assessed = [(lvl, score) for _, lvl, score in labeled if score is not None and lvl in (1, 3)]
    ranking = binary_ranking_metrics(
        [1 if lvl == 3 else 0 for lvl, _ in assessed],
        [score for _, score in assessed],
        max_fpr=GATE_MAX_FPR,
        threshold=threshold,
    )
    sweep = []
    for cut in SWEEP:
        entry = {"threshold": cut}
        for level in (1, 2, 3):
            scores = [score for _, lvl, score in labeled if lvl == level]
            entry[level] = sum(1 for score in scores if score is not None and score >= cut)
        sweep.append(entry)
    positives, negatives = per_level[3]["n"], per_level[1]["n"]
    recall = per_level[3]["flagged"] / positives if positives else None
    fpr = per_level[1]["flagged"] / negatives if negatives else None
    if positives < GATE_MIN_POSITIVES or negatives == 0:
        verdict = f"TIDAK DAPAT DINILAI (foto level 3 kurang dari {GATE_MIN_POSITIVES}, atau tanpa level 1)"
    elif recall >= GATE_MIN_RECALL and fpr <= GATE_MAX_FPR:
        verdict = "LULUS"
    else:
        verdict = "BELUM LULUS"
    return {
        "threshold": threshold,
        "rows_total": len(rows),
        "rows_labeled": len(labeled),
        "per_level": per_level,
        "ranking": ranking,
        "sweep": sweep,
        "recall": recall,
        "fpr": fpr,
        "verdict": verdict,
    }


def render(result: dict[str, Any]) -> str:
    per_level = result["per_level"]
    level_rows = [
        [f"level {level}", v["n"], v["not_assessed"], v["flagged"],
         _proportion(v["flagged"], v["n"])]
        for level, v in sorted(per_level.items())
    ]
    ranking = result["ranking"]
    summary = [
        ["ambang (dibekukan)", result["threshold"]],
        ["baris di CSV / berlabel", f"{result['rows_total']} / {result['rows_labeled']}"],
        ["recall level 3 (tidak dinilai dihitung terlewat)", _proportion(per_level[3]["flagged"], per_level[3]["n"])],
        ["level 1 salah ditandai cacat", _proportion(per_level[1]["flagged"], per_level[1]["n"])],
        ["AUC level 1 vs 3 (hanya yang dinilai)", ranking["auc"]],
    ]
    sweep_rows = [
        [entry["threshold"], _proportion(entry[3], per_level[3]["n"]),
         _proportion(entry[1], per_level[1]["n"]), _proportion(entry[2], per_level[2]["n"])]
        for entry in result["sweep"]
    ]
    return "\n\n".join([
        "# Evaluasi uji webcam model cacat",
        "## Per level cacat sebenarnya\n\n"
        + render_table(["level", "n", "tidak dinilai", "ditandai cacat", "proporsi"], level_rows),
        "## Ringkasan\n\n" + render_table(["ukuran", "nilai"], summary),
        f"## Gerbang\n\nSyarat (ambang beku {result['threshold']}): recall level 3 >= {GATE_MIN_RECALL:.0%}, "
        f"level 1 salah ditandai <= {GATE_MAX_FPR:.0%}, dan foto level 3 minimal {GATE_MIN_POSITIVES}.\n"
        f"Hasil: **{result['verdict']}**.",
        "## Sapuan ambang (hanya informasi)\n\n"
        + render_table(["ambang", "level 3 ditandai", "level 1 ditandai", "level 2 ditandai"], sweep_rows)
        + "\n\nJangan memilih ambang dari tabel ini lalu menilainya pada data yang sama; ambang baru "
          "harus diuji pada foto webcam baru.",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m tools.eval_webcam_defect", description=__doc__.splitlines()[0])
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5, help="ambang beku (bawaan 0,5)")
    parser.add_argument("--out", type=Path, default=None, help="simpan laporan Markdown")
    args = parser.parse_args()
    result = evaluate(load_rows(args.csv), args.threshold)
    if result["rows_labeled"] == 0:
        raise SystemExit("Belum ada baris berlabel: isi kolom true_defect_level (1/2/3) di CSV.")
    text = render(result)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
