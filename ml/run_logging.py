"""Pencatatan satu run training: log teks, CSV per epoch, tabel metrik, dan laporan Markdown.

Semua yang ditulis ke disk di-flush per epoch, jadi log tetap ada walaupun sesi (misalnya
Google Colab) terputus di tengah training.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

HISTORY_COLUMNS = (
    "task",
    "phase",
    "epoch",
    "phase_epoch",
    "loss",
    "accuracy",
    "val_loss",
    "val_accuracy",
    "lr",
    "seconds",
)
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_TRACKED_PACKAGES = (
    "tensorflow",
    "keras",
    "numpy",
    "scikit-learn",
    "Pillow",
    "image-classifiers",
)


# --------------------------------------------------------------------------- tabel


def format_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != 0 and abs(value) < 1e-3:
            return f"{value:.1e}"
        return f"{value:.{digits}f}"
    return str(value)


def _is_numeric(value: Any) -> bool:
    return value is None or (
        isinstance(value, (int, float)) and not isinstance(value, bool)
    )


def render_table(
    headers: Sequence[str], rows: Sequence[Sequence[Any]], digits: int = 4
) -> str:
    """Tabel Markdown yang juga rapi dibaca di terminal (kolom diratakan dengan spasi)."""
    cells = [
        [format_value(value, digits).replace("|", "\\|") for value in row]
        for row in rows
    ]
    header_cells = [str(header).replace("|", "\\|") for header in headers]
    columns = len(header_cells)
    for row in cells:
        if len(row) != columns:
            raise ValueError("Every row must have as many cells as there are headers.")
    right = [
        bool(rows) and all(_is_numeric(row[index]) for row in rows)
        for index in range(columns)
    ]
    widths = [
        max(3, len(header_cells[index]), *(len(row[index]) for row in cells))
        for index in range(columns)
    ]

    def line(values: Sequence[str]) -> str:
        padded = [
            values[index].rjust(widths[index])
            if right[index]
            else values[index].ljust(widths[index])
            for index in range(columns)
        ]
        return "| " + " | ".join(padded) + " |"

    separator = (
        "|"
        + "|".join(
            ("-" * (widths[index] + 1) + ":") if right[index] else ("-" * (widths[index] + 2))
            for index in range(columns)
        )
        + "|"
    )
    return "\n".join([line(header_cells), separator, *(line(row) for row in cells)])


def class_metrics_table(metrics: Mapping[str, Any]) -> str:
    """Precision, recall, F1, dan support per kelas dari ml.metrics.classification_metrics."""
    report = metrics["classification_report"]
    rows: list[list[Any]] = []
    total_support = 0
    for name in metrics["classes"]:
        entry = report[name]
        total_support += int(entry["support"])
        rows.append(
            [name, entry["precision"], entry["recall"], entry["f1-score"], int(entry["support"])]
        )
    rows.append(["accuracy", None, None, float(metrics["accuracy"]), total_support])
    for key in ("macro avg", "weighted avg"):
        if key in report:
            entry = report[key]
            rows.append(
                [key, entry["precision"], entry["recall"], entry["f1-score"], int(entry["support"])]
            )
    return render_table(["kelas", "precision", "recall", "f1", "support"], rows)


def confusion_table(metrics: Mapping[str, Any]) -> str:
    """Confusion matrix (baris = label sebenarnya, kolom = prediksi) plus recall per baris."""
    classes = list(metrics["classes"])
    matrix = metrics["confusion_matrix"]
    rows: list[list[Any]] = []
    for index, name in enumerate(classes):
        total = sum(matrix[index])
        rows.append(
            [name, *matrix[index], (matrix[index][index] / total) if total else None]
        )
    return render_table(["sebenarnya \\ prediksi", *classes, "recall"], rows)


def history_table(rows: Sequence[Mapping[str, Any]]) -> str:
    table_rows = [
        [
            row["phase"],
            int(row["epoch"]),
            row.get("loss"),
            row.get("accuracy"),
            row.get("val_loss"),
            row.get("val_accuracy"),
            row.get("lr"),
            row.get("seconds"),
        ]
        for row in rows
    ]
    return render_table(
        ["fase", "epoch", "loss", "acc", "val_loss", "val_acc", "lr", "detik"],
        table_rows,
        digits=4,
    )


def metric_sections(prefix: str, metrics: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        (f"{prefix} - metrik per kelas", class_metrics_table(metrics)),
        (f"{prefix} - confusion matrix", confusion_table(metrics)),
    ]


# --------------------------------------------------------------------------- history


def read_history(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    rows: list[dict[str, Any]] = []
    with Path(path).open(newline="", encoding="utf-8") as file:
        for raw in csv.DictReader(file):
            row: dict[str, Any] = {"task": raw["task"], "phase": raw["phase"]}
            for key in HISTORY_COLUMNS[2:]:
                text = raw.get(key, "")
                row[key] = float(text) if text not in ("", None) else None
            rows.append(row)
    return rows


class EpochRecorder:
    """Tulis satu baris CSV per epoch dan langsung flush ke disk (aman bila sesi putus)."""

    def __init__(
        self, history_path: Path, task: str, phase: str, epoch_offset: int = 0
    ) -> None:
        self.history_path = Path(history_path)
        self.task = task
        self.phase = phase
        self.epoch_offset = epoch_offset

    def record(
        self,
        phase_epoch: int,
        logs: Mapping[str, Any],
        lr: float | None = None,
        seconds: float | None = None,
    ) -> dict[str, Any]:
        def number(key: str) -> float | None:
            value = logs.get(key)
            return float(value) if value is not None else None

        row: dict[str, Any] = {
            "task": self.task,
            "phase": self.phase,
            "epoch": self.epoch_offset + phase_epoch,
            "phase_epoch": phase_epoch,
            "loss": number("loss"),
            "accuracy": number("accuracy"),
            "val_loss": number("val_loss"),
            "val_accuracy": number("val_accuracy"),
            "lr": lr,
            "seconds": seconds,
        }
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.history_path.is_file() or self.history_path.stat().st_size == 0
        with self.history_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=HISTORY_COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow({key: ("" if value is None else value) for key, value in row.items()})
            file.flush()
            os.fsync(file.fileno())
        logger.info(
            "[%s/%s] epoch %d | loss %s acc %s | val_loss %s val_acc %s | lr %s | %ss",
            self.task,
            self.phase,
            row["epoch"],
            format_value(row["loss"]),
            format_value(row["accuracy"]),
            format_value(row["val_loss"]),
            format_value(row["val_accuracy"]),
            format_value(row["lr"]),
            format_value(row["seconds"], 1),
        )
        return row


def make_keras_callback(recorder: EpochRecorder) -> Any:
    """Adapter tipis: meneruskan akhir tiap epoch Keras ke EpochRecorder."""
    import tensorflow as tf

    class _EpochLogCallback(tf.keras.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()
            self._started = time.perf_counter()

        def on_epoch_begin(self, epoch: int, logs: Any = None) -> None:
            self._started = time.perf_counter()

        def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
            learning_rate: float | None
            try:
                value = self.model.optimizer.learning_rate
                learning_rate = float(value.numpy()) if hasattr(value, "numpy") else float(value)
            except Exception:  # noqa: BLE001 - jadwal LR bukan angka tunggal; kolom dikosongkan
                learning_rate = None
            recorder.record(
                epoch + 1,
                logs or {},
                learning_rate,
                time.perf_counter() - self._started,
            )

    return _EpochLogCallback()


# --------------------------------------------------------------------------- run


def _git_info() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    info: dict[str, Any] = {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=root, check=False,
        )
        if commit.returncode == 0:
            info["commit"] = commit.stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5, cwd=root, check=False,
            )
            info["dirty"] = bool(status.stdout.strip()) if status.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def environment_info() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "git": _git_info(),
    }


class RunLog:
    """Satu folder run: train.log, history.csv, config.json, metrics.json, report.md."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._handlers: list[logging.Handler] = []
        self._previous_level: int | None = None

    @property
    def log_path(self) -> Path:
        return self.directory / "train.log"

    @property
    def history_path(self) -> Path:
        return self.directory / "history.csv"

    @property
    def report_path(self) -> Path:
        return self.directory / "report.md"

    @property
    def metrics_path(self) -> Path:
        return self.directory / "metrics.json"

    def attach(self, console: bool = True) -> None:
        """Alirkan semua log (logger.info/warning di seluruh paket ml) ke train.log dan layar."""
        if self._handlers:
            return
        root = logging.getLogger()
        self._previous_level = root.level
        if root.level == logging.NOTSET or root.level > logging.INFO:
            root.setLevel(logging.INFO)
        file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        self._handlers.append(file_handler)
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter("%(message)s"))
            self._handlers.append(console_handler)
        for handler in self._handlers:
            root.addHandler(handler)

    def close(self) -> None:
        root = logging.getLogger()
        for handler in self._handlers:
            root.removeHandler(handler)
            handler.close()
        self._handlers = []
        if self._previous_level is not None:
            root.setLevel(self._previous_level)
            self._previous_level = None

    def write_config(self, config: Mapping[str, Any]) -> None:
        payload = {**config, "environment": environment_info()}
        (self.directory / "config.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def write_metrics(self, metrics: Mapping[str, Any]) -> None:
        self.metrics_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def write_report(self, title: str, sections: Sequence[tuple[str, str]]) -> None:
        parts = [f"# {title}", ""]
        for heading, body in sections:
            parts += [f"## {heading}", "", body, ""]
        self.report_path.write_text("\n".join(parts), encoding="utf-8")
        logger.info("Laporan ditulis: %s", self.report_path)


def start_run(
    logs_dir: Path,
    run_name: str,
    architecture: str,
    config: Mapping[str, Any],
    console: bool = True,
) -> RunLog:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(logs_dir) / f"{run_name}_{architecture}_{stamp}"
    directory, counter = base, 1
    while directory.exists():
        counter += 1
        directory = Path(f"{base}-{counter}")
    run = RunLog(directory)
    run.attach(console=console)
    run.write_config(config)
    logger.info("Run log: %s", run.directory)
    return run


# --------------------------------------------------------------------------- grafik


def _pyplot() -> Any:
    """pyplot dengan backend Agg, atau None bila matplotlib tidak terpasang.

    MPLBACKEND bawaan Colab/Jupyter (module://matplotlib_inline...) tidak valid di luar notebook
    (mis. virtualenv terpisah) dan membuat `import matplotlib` melempar ValueError, jadi dipaksa Agg.
    """
    os.environ["MPLBACKEND"] = "Agg"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib tidak terpasang; grafik dilewati (pip install matplotlib).")
        return None
    return plt


def plot_curves(history_path: Path, output_dir: Path) -> list[Path]:
    """Kurva loss dan akurasi (train vs val) per task. Gagal/tanpa matplotlib: dilewati, tidak error."""
    plt = _pyplot()
    if plt is None:
        return []
    try:
        return _draw_curves(plt, history_path, output_dir)
    except Exception as exc:  # noqa: BLE001 - grafik hanya pelengkap; tidak boleh menggagalkan training
        logger.warning("Grafik kurva dilewati: %s", exc)
        return []


def _draw_curves(plt: Any, history_path: Path, output_dir: Path) -> list[Path]:
    rows = read_history(history_path)
    written: list[Path] = []
    for task in dict.fromkeys(row["task"] for row in rows):
        task_rows = [row for row in rows if row["task"] == task]
        epochs = [row["epoch"] for row in task_rows]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        for axis, (train_key, val_key, title) in zip(
            axes,
            (("loss", "val_loss", "Loss"), ("accuracy", "val_accuracy", "Akurasi")),
        ):
            axis.plot(epochs, [row[train_key] for row in task_rows], marker="o", label="train")
            axis.plot(epochs, [row[val_key] for row in task_rows], marker="o", label="validasi")
            switch = next(
                (row["epoch"] for row in task_rows if row["phase"] != task_rows[0]["phase"]),
                None,
            )
            if switch is not None:
                axis.axvline(switch - 0.5, color="gray", linestyle="--", label="mulai fine-tune")
            axis.set_title(f"{title} - {task}")
            axis.set_xlabel("epoch")
            axis.legend()
            axis.grid(alpha=0.3)
        path = Path(output_dir) / f"curves_{task}.png"
        figure.tight_layout()
        figure.savefig(path, dpi=130)
        plt.close(figure)
        written.append(path)
    return written


def plot_confusion(metrics: Mapping[str, Any], output_path: Path, title: str) -> Path | None:
    plt = _pyplot()
    if plt is None:
        return None
    try:
        return _draw_confusion(plt, metrics, output_path, title)
    except Exception as exc:  # noqa: BLE001 - lihat plot_curves
        logger.warning("Grafik confusion matrix dilewati: %s", exc)
        return None


def _draw_confusion(plt: Any, metrics: Mapping[str, Any], output_path: Path, title: str) -> Path:
    classes = list(metrics["classes"])
    matrix = metrics["confusion_matrix"]
    figure, axis = plt.subplots(figsize=(1.6 + 1.2 * len(classes), 1.4 + 1.1 * len(classes)))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(classes)), labels=classes)
    axis.set_yticks(range(len(classes)), labels=classes)
    axis.set_xlabel("prediksi")
    axis.set_ylabel("sebenarnya")
    axis.set_title(title)
    peak = max((max(row) for row in matrix), default=0)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            axis.text(j, i, str(value), ha="center", va="center",
                      color="white" if peak and value > peak / 2 else "black")
    figure.tight_layout()
    figure.savefig(output_path, dpi=130)
    plt.close(figure)
    return Path(output_path)
