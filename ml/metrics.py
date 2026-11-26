"""Metric helpers shared by training, evaluation, and model comparison.

Semua arsitektur (ResNeXt-50 dan ResNeXt-101) memakai fungsi yang sama di sini
sehingga angka keduanya dihitung dengan cara yang identik.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

GRADE_CLASSES = ("A", "B", "C")


def classification_metrics(
    expected: Sequence[int],
    predicted: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, Any]:
    labels = list(range(len(class_names)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        expected, predicted, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(f1_score(expected, predicted, average="macro")),
        "classes": list(class_names),
        "precision_per_class": {
            name: float(precision[index]) for index, name in enumerate(class_names)
        },
        "recall_per_class": {
            name: float(recall[index]) for index, name in enumerate(class_names)
        },
        "f1_per_class": {
            name: float(f1[index]) for index, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(
            expected, predicted, labels=labels
        ).tolist(),
        "classification_report": classification_report(
            expected,
            predicted,
            labels=labels,
            target_names=list(class_names),
            output_dict=True,
            zero_division=0,
        ),
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """Rentang kepercayaan 95% (Wilson) untuk proporsi; None bila total 0. Lebih jujur daripada p +- galat pada n kecil."""
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, float(center - half)), min(1.0, float(center + half))


def binary_ranking_metrics(
    expected: Sequence[int],
    scores: Sequence[float],
    max_fpr: float = 0.10,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """AUC dan dua titik operasi untuk positif (1) vs negatif (0).

    - fixed_threshold: recall dan false positive rate pada ambang tetap (bawaan 0,5).
    - constrained: recall tertinggi dengan false positive rate <= max_fpr. Ambangnya dipilih
      dari data yang sama, jadi angkanya optimistis; pakai sebagai indikasi, bukan hasil final.
    """
    truth = np.asarray(expected, dtype=int)
    score = np.asarray(scores, dtype=float)
    if truth.shape != score.shape:
        raise ValueError("expected and scores must have the same length.")
    positives = int(truth.sum())
    negatives = int(truth.size - positives)

    def operating_point(cut: float) -> dict[str, Any]:
        predicted = score >= cut
        true_positive = int(np.sum(predicted & (truth == 1)))
        false_positive = int(np.sum(predicted & (truth == 0)))
        return {
            "threshold": None if not np.isfinite(cut) else float(cut),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "recall": (true_positive / positives) if positives else None,
            "false_positive_rate": (false_positive / negatives) if negatives else None,
        }

    result: dict[str, Any] = {
        "positives": positives,
        "negatives": negatives,
        "max_fpr": max_fpr,
        "auc": None,
        "fixed_threshold": operating_point(threshold),
        "constrained": None,
    }
    if positives and negatives:
        result["auc"] = float(roc_auc_score(truth, score))
        fpr, tpr, thresholds = roc_curve(truth, score)
        allowed = np.flatnonzero(fpr <= max_fpr)
        best = allowed[int(np.argmax(tpr[allowed]))]
        result["constrained"] = operating_point(float(thresholds[best]))
    return result


def grade_metrics(expected: Sequence[str], predicted: Sequence[str]) -> dict[str, Any]:
    true_indices = [GRADE_CLASSES.index(value) for value in expected]
    predicted_indices = [GRADE_CLASSES.index(value) for value in predicted]
    return classification_metrics(true_indices, predicted_indices, GRADE_CLASSES)


def summarize_latency(durations_ms: Iterable[float]) -> dict[str, float]:
    values = np.asarray(list(durations_ms), dtype=np.float64)
    if values.size == 0:
        raise ValueError("At least one latency measurement is required.")
    return {
        "runs": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def benchmark_latency(
    models: Sequence[Any],
    preprocess_input: Any,
    input_size: tuple[int, int],
    runs: int = 30,
    warmup: int = 3,
    seed: int = 0,
) -> dict[str, float]:
    """Time one full prediction (all models in sequence) on a fixed synthetic image, in ms."""
    if runs < 1 or warmup < 0:
        raise ValueError("runs must be >= 1 and warmup >= 0.")
    rng = np.random.default_rng(seed)
    pixels = rng.uniform(0, 255, size=(1, *input_size, 3)).astype(np.float32)
    batch = preprocess_input(pixels)
    durations: list[float] = []
    for attempt in range(warmup + runs):
        start = time.perf_counter()
        for model in models:
            model.predict(batch, verbose=0)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if attempt >= warmup:
            durations.append(elapsed_ms)
    return summarize_latency(durations)


def count_parameters(models: Sequence[Any]) -> int:
    return int(sum(model.count_params() for model in models))


def describe_device() -> str:
    """Return 'GPU: <name>' or 'CPU' for the runtime TensorFlow will use."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return "CPU"
    names = []
    for gpu in gpus:
        try:
            details = tf.config.experimental.get_device_details(gpu)
            names.append(str(details.get("device_name", gpu.name)))
        except Exception:  # noqa: BLE001 - device details are best-effort metadata
            names.append(gpu.name)
    return "GPU: " + ", ".join(names)
