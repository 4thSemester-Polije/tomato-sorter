"""Train and evaluate separate ripeness and defect classifiers (ResNeXt-50 or ResNeXt-101)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.model_selection import train_test_split

from ml.metrics import (
    benchmark_latency,
    binary_ranking_metrics,
    classification_metrics,
    count_parameters,
    describe_device,
    grade_metrics,
)
from ml.resnext import (
    ARCHITECTURES,
    DEFAULT_ARCHITECTURE,
    DEFECT_CLASSES,
    DEFECT_LEVELS,
    FINE_TUNE_LAYERS,
    INPUT_SIZE,
    RIPENESS_CLASSES,
    RIPENESS_LEVELS,
    artifact_paths,
    create_model,
    display_name,
    get_preprocess_input,
    load_trusted_model,
    report_path,
    validate_architecture,
)
from ml.rubric import grade_from_levels
from ml.run_logging import (
    EpochRecorder,
    RunLog,
    history_table,
    make_keras_callback,
    metric_sections,
    plot_confusion,
    plot_curves,
    read_history,
    render_table,
    start_run,
)

logger = logging.getLogger(__name__)

# Hyperparameter tetap untuk semua arsitektur. Nilai ini dicatat di metadata
# model agar ml.compare dapat memastikan perbandingan ResNeXt-50 vs 101 adil.
HEAD_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
EARLY_STOPPING_PATIENCE = 3

TASKS = ("ripeness", "defect")

# Gerbang go/no-go model cacat pada foto primer (lihat docs/TRAINING_CACAT.md).
GATE_MIN_RECALL = 0.70
GATE_MAX_FPR = 0.10
GATE_MIN_POSITIVES = 5  # di bawah ini, recall terlalu kasar untuk dijadikan keputusan


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str
    digest: str


def _collect_samples(class_directories: dict[str, Path]) -> list[Sample]:
    """Deduplicate identical files and exclude copies carrying conflicting labels."""
    by_digest: dict[str, list[Sample]] = {}
    for label, directory in class_directories.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset class directory not found: {directory}")
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
            }:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            by_digest.setdefault(digest, []).append(Sample(path, label, digest))

    if not by_digest:
        raise ValueError(f"No supported JPG/PNG images found in {class_directories}.")

    unique_samples: list[Sample] = []
    conflicting_groups: list[list[Sample]] = []
    for copies in by_digest.values():
        labels = {sample.label for sample in copies}
        if len(labels) > 1:
            conflicting_groups.append(copies)
        else:
            unique_samples.append(copies[0])

    if conflicting_groups:
        logger.warning(
            "Excluded %d identical-image group(s) with conflicting labels (%d files): %s",
            len(conflicting_groups),
            sum(len(group) for group in conflicting_groups),
            [
                [str(sample.path) for sample in group]
                for group in conflicting_groups
            ],
        )
    return sorted(unique_samples, key=lambda sample: str(sample.path))


def _split_samples(
    samples: Sequence[Sample],
    seed: int,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Stratify by class into 70/15/15 partitions after exact-image deduplication."""
    labels = [sample.label for sample in samples]
    counts = Counter(labels)
    if len(counts) < 2 or min(counts.values()) < 3:
        raise ValueError(
            "Each class needs at least three unique images for a stratified split; "
            f"found {dict(counts)}."
        )
    train_validation, test = train_test_split(
        list(samples),
        test_size=0.15,
        random_state=seed,
        stratify=labels,
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.15 / 0.85,
        random_state=seed,
        stratify=[sample.label for sample in train_validation],
    )
    return list(train), list(validation), list(test)


def _split_samples_from_manifest(
    samples: Sequence[Sample],
    manifest_path: str | Path,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Pakai split dari CSV (kolom: file, split, group) alih-alih split acak.

    `file` berbentuk "<kelas>/<nama file>". Semua anggota satu `group` (misalnya satu foto sumber
    atau satu video) wajib berada di split yang sama, supaya tidak ada kebocoran near-duplicate.
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    assignments: dict[str, str] = {}
    group_splits: dict[str, set[str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing_columns = {"file", "split", "group"} - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Split manifest {path} lacks column(s): {sorted(missing_columns)}."
            )
        for row in reader:
            split = row["split"].strip().lower()
            if split not in {"train", "validation", "test"}:
                raise ValueError(f"Unknown split {row['split']!r} in {path}.")
            assignments[row["file"].strip().replace("\\", "/")] = split
            group_splits.setdefault(row["group"].strip(), set()).add(split)

    leaking = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaking:
        raise ValueError(
            f"{len(leaking)} group(s) span more than one split (data leakage), "
            f"for example: {leaking[:5]}."
        )

    parts: dict[str, list[Sample]] = {"train": [], "validation": [], "test": []}
    unassigned: list[str] = []
    for sample in samples:
        key = f"{sample.label}/{sample.path.name}"
        split = assignments.get(key)
        if split is None:
            unassigned.append(key)
        else:
            parts[split].append(sample)
    if unassigned:
        raise ValueError(
            f"{len(unassigned)} image(s) are not listed in the split manifest, "
            f"for example: {unassigned[:5]}."
        )
    for name, part in parts.items():
        if len({sample.label for sample in part}) < 2:
            raise ValueError(f"Split {name!r} must contain both classes; found {len(part)} image(s).")
    return parts["train"], parts["validation"], parts["test"]


def _split_percentages(
    train: Sequence[Sample],
    validation: Sequence[Sample],
    test: Sequence[Sample],
    from_manifest: bool,
) -> dict[str, int]:
    if not from_manifest:
        return {"train": 70, "validation": 15, "test": 15}
    total = len(train) + len(validation) + len(test)
    return {
        "train": round(100 * len(train) / total),
        "validation": round(100 * len(validation) / total),
        "test": round(100 * len(test) / total),
    }


def _log_split_summary(
    task: str,
    train: Sequence[Sample],
    validation: Sequence[Sample],
    test: Sequence[Sample],
) -> None:
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        logger.info(
            "[%s] %s: %d gambar %s",
            task,
            name,
            len(part),
            dict(sorted(Counter(sample.label for sample in part).items())),
        )


def _epoch_callbacks(
    run: RunLog | None,
    task: str,
    phase: str,
    epoch_offset: int,
    tensorboard: bool,
) -> list[Any]:
    """Callback pencatat per epoch (CSV + log) dan, opsional, TensorBoard."""
    if run is None:
        return []
    callbacks: list[Any] = [
        make_keras_callback(EpochRecorder(run.history_path, task, phase, epoch_offset))
    ]
    if tensorboard and importlib.util.find_spec("tensorboard") is None:
        logger.warning("TensorBoard tidak terpasang; --tensorboard dilewati (pip install tensorboard).")
    elif tensorboard:
        import tensorflow as tf

        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=str(run.directory / "tensorboard" / f"{task}_{phase}")
            )
        )
    return callbacks


def _load_tensorflow_dataset(
    samples: Sequence[Sample],
    class_to_index: dict[str, int],
    preprocess_input: Any,
    batch_size: int,
    training: bool,
    seed: int = 42,
) -> Any:
    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices(
        ([str(sample.path) for sample in samples],
         [class_to_index[sample.label] for sample in samples])
    )

    def load_image(path: Any, label: Any) -> tuple[Any, Any]:
        image = tf.io.decode_image(
            tf.io.read_file(path),
            channels=3,
            expand_animations=False,
        )
        image.set_shape((None, None, 3))
        image = tf.image.resize(image, INPUT_SIZE)
        return preprocess_input(image), label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(
            len(samples),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _metric_report(
    model: Any,
    dataset: Any,
    class_names: Sequence[str],
) -> dict[str, Any]:
    expected = np.concatenate([labels.numpy() for _, labels in dataset])
    predicted = np.argmax(model.predict(dataset, verbose=0), axis=1)
    return classification_metrics(expected, predicted, class_names)


def _train_task(
    task: str,
    samples: list[Sample],
    class_names: Sequence[str],
    class_levels: dict[str, int],
    output_path: Path,
    seed: int,
    batch_size: int,
    head_epochs: int,
    fine_tune_epochs: int,
    reuse_existing: bool = False,
    architecture: str = DEFAULT_ARCHITECTURE,
    split_manifest: Path | None = None,
    run: RunLog | None = None,
    tensorboard: bool = False,
) -> tuple[Any, dict[str, Any]]:
    import tensorflow as tf

    validate_architecture(architecture)
    # Seed global (Python, NumPy, TensorFlow) agar tiap arsitektur mulai dari kondisi acak yang sama.
    tf.keras.utils.set_random_seed(seed)
    if split_manifest is not None:
        train_samples, validation_samples, test_samples = _split_samples_from_manifest(
            samples, split_manifest
        )
    else:
        train_samples, validation_samples, test_samples = _split_samples(samples, seed)
    _log_split_summary(task, train_samples, validation_samples, test_samples)
    class_to_index = {name: index for index, name in enumerate(class_names)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_reused = reuse_existing and output_path.is_file()
    if checkpoint_reused:
        logger.info("Reusing existing model checkpoint: %s", output_path)
        model = load_trusted_model(output_path)
        backbone = None
        preprocess_input = get_preprocess_input(architecture)
        history: dict[str, list[float]] = {}
    else:
        model, backbone, preprocess_input = create_model(
            task, len(class_names), architecture=architecture
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=HEAD_LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        history = {}
    train_data = _load_tensorflow_dataset(
        train_samples, class_to_index, preprocess_input, batch_size, True, seed
    )
    validation_data = _load_tensorflow_dataset(
        validation_samples, class_to_index, preprocess_input, batch_size, False, seed
    )
    test_data = _load_tensorflow_dataset(
        test_samples, class_to_index, preprocess_input, batch_size, False, seed
    )

    training_seconds: float | None = None
    if not checkpoint_reused:
        training_started = time.perf_counter()
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(output_path),
                monitor="val_loss",
                save_best_only=True,
            ),
        ]
        fit_history = model.fit(
            train_data,
            validation_data=validation_data,
            epochs=head_epochs,
            callbacks=callbacks + _epoch_callbacks(run, task, "head", 0, tensorboard),
            verbose=2,
        )
        history.update(
            {key: [float(value) for value in values] for key, values in fit_history.history.items()}
        )
        completed_epochs = len(fit_history.history.get("loss", []))
        if fine_tune_epochs > 0:
            backbone.trainable = True
            for layer in backbone.layers[:-FINE_TUNE_LAYERS]:
                layer.trainable = False
            for layer in backbone.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            fine_tune_history = model.fit(
                train_data,
                validation_data=validation_data,
                epochs=fine_tune_epochs,
                callbacks=callbacks
                + _epoch_callbacks(run, task, "fine_tune", completed_epochs, tensorboard),
                verbose=2,
            )
            history.update(
                {
                    f"fine_tune_{key}": [float(value) for value in values]
                    for key, values in fine_tune_history.history.items()
                }
            )
        training_seconds = time.perf_counter() - training_started

    metadata = {
        "model": display_name(architecture),
        "architecture": architecture,
        "task": task,
        "input_size": list(INPUT_SIZE),
        "class_names": list(class_names),
        "class_levels": class_levels,
        "pretrained_weights": "ImageNet",
        "seed": seed,
        "hyperparameters": {
            "seed": seed,
            "batch_size": batch_size,
            "head_epochs": head_epochs,
            "fine_tune_epochs": fine_tune_epochs,
            "head_learning_rate": HEAD_LEARNING_RATE,
            "fine_tune_learning_rate": FINE_TUNE_LEARNING_RATE,
            "fine_tune_layers": FINE_TUNE_LAYERS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "input_size": list(INPUT_SIZE),
        },
        "parameters": count_parameters([model]),
        "training_seconds": training_seconds,
        "split_percentages": _split_percentages(
            train_samples, validation_samples, test_samples, split_manifest is not None
        ),
        "split_source": "manifest" if split_manifest is not None else "stratified_random",
        "unique_images": {
            "total": len(samples),
            "train": len(train_samples),
            "validation": len(validation_samples),
            "test": len(test_samples),
        },
        "split_manifest": {
            "train": [
                {"sha256": sample.digest, "class": sample.label, "file": sample.path.name}
                for sample in train_samples
            ],
            "validation": [
                {
                    "sha256": sample.digest,
                    "class": sample.label,
                    "file": sample.path.name,
                }
                for sample in validation_samples
            ],
            "test": [
                {"sha256": sample.digest, "class": sample.label, "file": sample.path.name}
                for sample in test_samples
            ],
        },
        "test_metrics": _metric_report(model, test_data, class_names),
        "validation_metrics": _metric_report(model, validation_data, class_names),
        "history": history,
        "checkpoint_reused": checkpoint_reused,
    }
    return model, metadata


def _primary_test_samples(data_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    try:
        import pillow_heif
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError(
            "Primary HEIC evaluation requires pillow-heif and openpyxl."
        ) from exc

    pillow_heif.register_heif_opener()
    primary_root = data_root / "Primer"
    photo_directory = primary_root / "Foto"
    labels_path = primary_root / "Label" / "Labelv.1.xlsx"
    mapping_path = primary_root / "Label" / "mapping_nama_file.csv"
    for required_path in (photo_directory, labels_path, mapping_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Primary evaluation input not found: {required_path}")

    with mapping_path.open(newline="", encoding="utf-8-sig") as file:
        filename_by_id = {
            Path(row["nama_file_baru"]).stem: row["nama_file_baru"]
            for row in csv.DictReader(file)
        }

    worksheet = load_workbook(labels_path, read_only=True, data_only=True)["Pelabelan"]
    rows = worksheet.iter_rows(values_only=True)
    headers = next(rows)
    samples: list[tuple[Path, dict[str, Any]]] = []
    for values in rows:
        row = dict(zip(headers, values))
        tomato_id = row.get("id_tomat")
        if not tomato_id:
            continue
        filename = filename_by_id.get(str(tomato_id))
        if filename is None:
            raise ValueError(f"No mapped photo filename for tomato {tomato_id}.")
        path = photo_directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Primary photo not found: {path}")
        for column in ("tingkat_kematangan", "tingkat_cacat", "grade"):
            if row.get(column) is None:
                raise ValueError(f"Missing {column} label for tomato {tomato_id}.")
        ripeness_level = int(row["tingkat_kematangan"])
        defect_level = int(row["tingkat_cacat"])
        if ripeness_level not in (1, 2, 3) or defect_level not in (1, 2, 3):
            raise ValueError(f"Invalid rubric level for tomato {tomato_id}.")
        expected_grade = grade_from_levels(ripeness_level, defect_level)
        if str(row["grade"]).upper() != expected_grade:
            raise ValueError(
                f"Grade for tomato {tomato_id} conflicts with the rubric: "
                f"expected {expected_grade}, found {row['grade']!r}."
            )
        samples.append((path, row))
    if not samples:
        raise ValueError(f"No labelled primary photos found in {labels_path}.")
    return samples


def _evaluate_primary(
    data_root: Path,
    ripeness_model: Any,
    defect_model: Any,
    models_dir: Path,
    architecture: str = DEFAULT_ARCHITECTURE,
) -> dict[str, Any]:
    from PIL import Image

    ripeness_metadata = json.loads(
        artifact_paths(models_dir, "ripeness", architecture)[1].read_text(encoding="utf-8")
    )
    defect_metadata = json.loads(
        artifact_paths(models_dir, "defect", architecture)[1].read_text(encoding="utf-8")
    )
    preprocess_input = get_preprocess_input(architecture)

    expected_grades: list[str] = []
    predicted_grades: list[str] = []
    ripeness_true: list[int] = []
    ripeness_predicted: list[int] = []
    defect_true: list[int] = []
    defect_predicted: list[int] = []
    primary_samples = _primary_test_samples(data_root)

    for path, labels in primary_samples:
        with Image.open(path) as source:
            image = source.convert("RGB").resize((INPUT_SIZE[1], INPUT_SIZE[0]))
            pixels = np.asarray(image, dtype=np.float32)
        processed = preprocess_input(np.expand_dims(pixels, axis=0))
        ripeness_probabilities = ripeness_model.predict(processed, verbose=0)[0]
        defect_probabilities = defect_model.predict(processed, verbose=0)[0]
        ripeness_classes = ripeness_metadata["class_names"]
        defect_classes = defect_metadata["class_names"]
        ripeness_class = ripeness_classes[int(np.argmax(ripeness_probabilities))]
        defect_class = defect_classes[int(np.argmax(defect_probabilities))]
        ripeness_level = RIPENESS_LEVELS[ripeness_class]
        defect_level = DEFECT_LEVELS[defect_class]
        expected_grades.append(str(labels["grade"]).upper())
        predicted_grades.append(grade_from_levels(ripeness_level, defect_level))
        ripeness_true.append(int(labels["tingkat_kematangan"]) - 1)
        ripeness_predicted.append(ripeness_classes.index(ripeness_class))
        if int(labels["tingkat_cacat"]) in (1, 3):
            defect_true.append(0 if int(labels["tingkat_cacat"]) == 1 else 1)
            defect_predicted.append(defect_classes.index(defect_class))

    grade_two_count = sum(
        1 for _, labels in primary_samples if int(labels["tingkat_cacat"]) == 2
    )
    return {
        "independent_primary_test_images": len(expected_grades),
        "grade": grade_metrics(expected_grades, predicted_grades),
        "heads": {
            "ripeness": classification_metrics(
                ripeness_true, ripeness_predicted, RIPENESS_CLASSES
            ),
            "defect": {
                **classification_metrics(
                    defect_true, defect_predicted, DEFECT_CLASSES
                ),
                "evaluated_images_with_binary_ground_truth": len(defect_true),
                "excluded_level_2_ground_truth_images": grade_two_count,
            },
        },
    }


def _crop_for_inference(image: Any, mode: str) -> Any:
    """Crop tomat lewat jalur yang sama dengan API (vision.crop.crop_for_inference), atau None."""
    from vision.crop import crop_for_inference

    return crop_for_inference(image, mode)


def _evaluate_defect_primary(
    data_root: Path,
    defect_model: Any,
    class_names: Sequence[str],
    architecture: str = DEFAULT_ARCHITECTURE,
    crop_mode: str | None = None,
) -> dict[str, Any]:
    """Evaluasi khusus model cacat pada foto primer, termasuk p_reject per foto.

    crop_mode None = foto penuh (seperti model lama); "neutral"/"keep" = crop tomat dengan latar
    netral/asli, harus sama dengan cara dataset training dibuat. Foto yang crop-nya gagal tidak ikut
    dihitung dan dilaporkan terpisah.
    """
    from PIL import Image

    preprocess_input = get_preprocess_input(architecture)
    reject_index = list(class_names).index("reject")
    rows: list[dict[str, Any]] = []
    for path, labels in _primary_test_samples(data_root):
        with Image.open(path) as source:
            image = source.convert("RGB")
        status = "full_frame"
        if crop_mode is not None:
            cropped = _crop_for_inference(image, crop_mode)
            if cropped is None:
                rows.append(
                    {
                        "file": path.name,
                        "actual_defect_level": int(labels["tingkat_cacat"]),
                        "p_reject": None,
                        "predicted": None,
                        "crop_status": "gagal",
                    }
                )
                continue
            image, status = cropped, f"crop_{crop_mode}"
        pixels = np.asarray(image.resize((INPUT_SIZE[1], INPUT_SIZE[0])), dtype=np.float32)
        probabilities = defect_model.predict(
            preprocess_input(np.expand_dims(pixels, axis=0)), verbose=0
        )[0]
        rows.append(
            {
                "file": path.name,
                "actual_defect_level": int(labels["tingkat_cacat"]),
                "p_reject": float(probabilities[reject_index]),
                "predicted": class_names[int(np.argmax(probabilities))],
                "crop_status": status,
            }
        )

    scored = [row for row in rows if row["p_reject"] is not None]
    per_level: dict[int, dict[str, Any]] = {}
    for level in (1, 2, 3):
        scores = [row["p_reject"] for row in scored if row["actual_defect_level"] == level]
        rejected = sum(
            1 for row in scored
            if row["actual_defect_level"] == level and row["predicted"] == "reject"
        )
        per_level[level] = {
            "images": len(scores),
            "no_crop": sum(
                1 for row in rows
                if row["actual_defect_level"] == level and row["p_reject"] is None
            ),
            "predicted_reject": rejected,
            "predicted_reject_rate": (rejected / len(scores)) if scores else None,
            "mean_p_reject": float(np.mean(scores)) if scores else None,
            "median_p_reject": float(np.median(scores)) if scores else None,
        }

    binary = [row for row in scored if row["actual_defect_level"] in (1, 3)]
    level_1_vs_3 = binary_ranking_metrics(
        [1 if row["actual_defect_level"] == 3 else 0 for row in binary],
        [row["p_reject"] for row in binary],
        max_fpr=GATE_MAX_FPR,
    )
    level_1_vs_2_or_3 = binary_ranking_metrics(
        [0 if row["actual_defect_level"] == 1 else 1 for row in scored],
        [row["p_reject"] for row in scored],
        max_fpr=GATE_MAX_FPR,
    )
    constrained = level_1_vs_3["constrained"]
    enough_positives = level_1_vs_3["positives"] >= GATE_MIN_POSITIVES
    passed = bool(
        enough_positives
        and constrained is not None
        and constrained["recall"] is not None
        and constrained["recall"] >= GATE_MIN_RECALL
    )
    return {
        "images": len(rows),
        "scored_images": len(scored),
        "no_crop": len(rows) - len(scored),
        "input": "full_frame" if crop_mode is None else f"crop_{crop_mode}",
        "per_level": per_level,
        "level1_vs_level3": level_1_vs_3,
        "level1_vs_level2_or_3": level_1_vs_2_or_3,
        "gate": {
            "min_recall": GATE_MIN_RECALL,
            "max_fpr": GATE_MAX_FPR,
            "min_positives": GATE_MIN_POSITIVES,
            "enough_positives": enough_positives,
            "passed": passed,
        },
        "rows": rows,
    }


def _defect_primary_sections(primary: dict[str, Any]) -> list[tuple[str, str]]:
    level_rows = [
        [
            f"level {level}",
            values["images"],
            values.get("no_crop", 0),
            values["predicted_reject"],
            values["predicted_reject_rate"],
            values["mean_p_reject"],
            values["median_p_reject"],
        ]
        for level, values in sorted(primary["per_level"].items())
    ]
    ranking = primary["level1_vs_level3"]
    fixed = ranking["fixed_threshold"]
    constrained = ranking["constrained"] or {}
    summary_rows = [
        ["masukan model", None, str(primary.get("input", "full_frame"))],
        ["foto tanpa crop valid (tidak dihitung)", primary.get("no_crop", 0), ""],
        ["jumlah level 1 / level 3", None, f"{ranking['negatives']} / {ranking['positives']}"],
        ["AUC level 1 vs 3", ranking["auc"], ""],
        ["AUC level 1 vs 2+3", primary["level1_vs_level2_or_3"]["auc"], ""],
        ["recall level 3 (ambang 0,5)", fixed["recall"], ""],
        ["false positive rate level 1 (ambang 0,5)", fixed["false_positive_rate"], ""],
        [
            f"recall level 3 @ FPR <= {ranking['max_fpr']:.0%}",
            constrained.get("recall"),
            f"ambang {format_threshold(constrained.get('threshold'))}",
        ],
    ]
    gate = primary["gate"]
    if not gate["enough_positives"]:
        verdict = f"TIDAK DAPAT DINILAI (foto level 3 kurang dari {gate['min_positives']})"
    else:
        verdict = "LULUS" if gate["passed"] else "BELUM LULUS"
    gate_text = (
        f"Syarat: recall level 3 >= {gate['min_recall']:.0%} pada false positive <= "
        f"{gate['max_fpr']:.0%} di foto primer.\n"
        f"Hasil: **{verdict}**.\n\n"
        "Catatan: ambang dipilih dari data yang sama sehingga angkanya optimistis, dan jumlah "
        f"positif hanya {ranking['positives']} foto, jadi anggap ini indikasi kasar."
    )
    return [
        (
            "Primer - model cacat per level cacat sebenarnya",
            render_table(
                ["level", "n", "tanpa crop", "diprediksi reject", "proporsi", "mean p_reject",
                 "median p_reject"],
                level_rows,
            ),
        ),
        (
            "Primer - ringkasan model cacat",
            render_table(["ukuran", "nilai", "keterangan"], summary_rows),
        ),
        ("Primer - gate go/no-go", gate_text),
    ]


def format_threshold(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _read_preparation_info(directory: Path | None) -> dict[str, Any] | None:
    """Parameter persiapan dataset (prepare_report.json dari ml.prepare_defect_v2), bila ada."""
    if directory is None:
        return None
    path = Path(directory) / "prepare_report.json"
    if not path.is_file():
        return None
    try:
        parameters = json.loads(path.read_text(encoding="utf-8")).get("parameters")
    except (OSError, ValueError):
        return None
    return parameters if isinstance(parameters, dict) else None


def _resolve_primary_crop(
    primary_crop: str | None, preparation: dict[str, Any] | None
) -> str | None:
    """auto -> ikuti cara dataset dibuat; selain itu pakai pilihan eksplisit dan peringatkan bila beda."""
    if primary_crop == "auto":
        resolved = preparation.get("background") if preparation else None
        logger.info("--primary-crop auto -> %s", resolved or "none (foto penuh)")
        return resolved
    built_with = preparation.get("background") if preparation else None
    if built_with and primary_crop != built_with:
        logger.warning(
            "Dataset dibuat dengan crop latar '%s' tetapi evaluasi primer memakai '%s'. "
            "Hasil gate tidak sebanding; gunakan --primary-crop %s.",
            built_with,
            primary_crop or "foto penuh",
            built_with,
        )
    return primary_crop


def _split_count_table(metadata: dict[str, Any]) -> str:
    classes = metadata["class_names"]
    rows = []
    for split in ("train", "validation", "test"):
        counts = Counter(entry["class"] for entry in metadata["split_manifest"][split])
        rows.append([split, sum(counts.values()), *(counts.get(name, 0) for name in classes)])
    return render_table(["split", "total", *classes], rows)


def _write_run_outputs(
    run: RunLog,
    selected: Sequence[str],
    metadata: dict[str, dict[str, Any]],
    report: dict[str, Any],
    primary_defect: dict[str, Any] | None,
    primary_error: str | None,
) -> None:
    """Tulis report.md (tabel), metrics.json, predictions_primer.csv, dan grafik ke folder run."""
    history = read_history(run.history_path)
    sections: list[tuple[str, str]] = []
    for task in selected:
        task_history = [row for row in history if row["task"] == task]
        if task_history:
            sections.append((f"Epoch - {task}", history_table(task_history)))
        sections.append((f"Data - {task}", _split_count_table(metadata[task])))
        if "validation_metrics" in metadata[task]:
            sections += metric_sections(
                f"Validasi sekunder ({task})", metadata[task]["validation_metrics"]
            )
        sections += metric_sections(f"Test sekunder ({task})", metadata[task]["test_metrics"])

    primary = report.get("primary_independent_test")
    if primary:
        sections += metric_sections("Primer - grade", primary["grade"])
        sections += metric_sections("Primer - head kematangan", primary["heads"]["ripeness"])
        sections += metric_sections(
            "Primer - head cacat (level 1 vs 3)", primary["heads"]["defect"]
        )
    if primary_defect is not None:
        sections += _defect_primary_sections(primary_defect)
    if primary_error is not None:
        sections.append(
            (
                "Primer - evaluasi gagal",
                f"Model sudah tersimpan, tetapi evaluasi foto primer gagal: {primary_error}",
            )
        )

    for heading, body in sections:
        logger.info("\n## %s\n%s", heading, body)
    run.write_report(f"Laporan training {', '.join(selected)} ({report['model']})", sections)
    run.write_metrics(
        {
            "report": report,
            "validation": {
                task: metadata[task].get("validation_metrics") for task in selected
            },
            "defect_primary": primary_defect,
        }
    )
    if primary_defect is not None:
        with (run.directory / "predictions_primer.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["file", "actual_defect_level", "p_reject", "predicted", "crop_status"],
                restval="",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(primary_defect["rows"])
    plot_curves(run.history_path, run.directory)
    for task in selected:
        plot_confusion(
            metadata[task]["test_metrics"],
            run.directory / f"confusion_test_{task}.png",
            f"Test sekunder - {task}",
        )


def _run_training(
    root: Path,
    output_directory: Path,
    selected: Sequence[str],
    seed: int,
    batch_size: int,
    head_epochs: int,
    fine_tune_epochs: int,
    reuse_existing: bool,
    architecture: str,
    defect_data_dir: Path | None,
    split_manifest: Path | None,
    run: RunLog | None,
    tensorboard: bool,
    primary_crop: str | None = None,
) -> dict[str, Any]:
    secondary = root / "Sekunder"
    ripeness_root = secondary / "kematangan" / "Klasifikasi Kematangan Tomat"
    ripeness_directories = {
        "matang": ripeness_root / "Tomat Matang",
        "setengah_matang": ripeness_root / "Tomat Setengah Matang",
        "mentah": ripeness_root / "Tomat Mentah",
    }
    defect_root = (
        defect_data_dir if defect_data_dir is not None else secondary / "cacat" / "Two Classes"
    )
    defect_directories = (
        {"healthy": defect_root / "healthy", "reject": defect_root / "reject"}
        if defect_data_dir is not None
        else {"healthy": defect_root / "Healthy", "reject": defect_root / "Reject"}
    )
    definitions = {
        "ripeness": (RIPENESS_CLASSES, RIPENESS_LEVELS, ripeness_directories),
        "defect": (DEFECT_CLASSES, DEFECT_LEVELS, defect_directories),
    }

    preparation = _read_preparation_info(defect_data_dir) if "defect" in selected else None
    primary_crop = _resolve_primary_crop(primary_crop, preparation)

    # Kumpulkan semua sampel dulu supaya folder yang salah ketahuan sebelum training mulai.
    collected = {task: _collect_samples(definitions[task][2]) for task in selected}
    models: dict[str, Any] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for task in selected:
        class_names, class_levels, _ = definitions[task]
        models[task], metadata[task] = _train_task(
            task,
            collected[task],
            class_names,
            class_levels,
            artifact_paths(output_directory, task, architecture)[0],
            seed,
            batch_size,
            head_epochs,
            fine_tune_epochs,
            reuse_existing,
            architecture,
            split_manifest,
            run,
            tensorboard,
        )

    if preparation is not None and "defect" in metadata:
        metadata["defect"]["data_preparation"] = preparation
    for task in selected:
        artifact_paths(output_directory, task, architecture)[1].write_text(
            json.dumps(metadata[task], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    report: dict[str, Any] = {
        "model": display_name(architecture),
        "architecture": architecture,
        "input_size": list(INPUT_SIZE),
        "seed": seed,
        "tasks": list(selected),
        "secondary": {task: metadata[task]["test_metrics"] for task in selected},
    }
    primary_defect: dict[str, Any] | None = None
    primary_error: str | None = None
    if set(selected) == set(TASKS):
        report["primary_independent_test"] = _evaluate_primary(
            root, models["ripeness"], models["defect"], output_directory, architecture
        )
        report["efficiency"] = measure_efficiency(
            models["ripeness"], models["defect"], architecture
        )
        report_path(output_directory, architecture).write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    elif "defect" in selected:
        # Model sudah tersimpan; kegagalan evaluasi primer tidak boleh menghapus hasil training.
        try:
            primary_defect = _evaluate_defect_primary(
                root, models["defect"], DEFECT_CLASSES, architecture, primary_crop
            )
            report["defect_primary_independent_test"] = {
                key: value for key, value in primary_defect.items() if key != "rows"
            }
        except Exception as exc:  # noqa: BLE001 - lihat komentar di atas
            primary_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Evaluasi foto primer model cacat gagal.")
    if run is not None:
        _write_run_outputs(run, selected, metadata, report, primary_defect, primary_error)
    return report


def train(
    data_root: str | Path,
    models_dir: str | Path = "models",
    seed: int = 42,
    batch_size: int = 16,
    head_epochs: int = 8,
    fine_tune_epochs: int = 5,
    reuse_existing: bool = False,
    architecture: str = DEFAULT_ARCHITECTURE,
    tasks: Sequence[str] = TASKS,
    defect_data_dir: str | Path | None = None,
    split_manifest: str | Path | None = None,
    logs_dir: str | Path | None = None,
    run_name: str = "train",
    tensorboard: bool = False,
    overwrite_models: bool = False,
    primary_crop: str | None = None,
) -> dict[str, Any]:
    """Train the requested heads and evaluate on secondary holdouts and primary photos."""
    validate_architecture(architecture)
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero.")
    if head_epochs < 1:
        raise ValueError("head_epochs must be greater than zero.")
    if fine_tune_epochs < 0:
        raise ValueError("fine_tune_epochs cannot be negative.")
    selected = list(dict.fromkeys(tasks))
    if not selected or any(task not in TASKS for task in selected):
        raise ValueError(f"tasks must be a non-empty subset of {TASKS}; got {list(tasks)}.")
    if split_manifest is not None and len(selected) != 1:
        raise ValueError("split_manifest can only be used when training a single task.")
    if primary_crop not in (None, "auto", "neutral", "keep"):
        raise ValueError("primary_crop must be None, 'auto', 'neutral' or 'keep'.")
    root = Path(data_root)
    output_directory = Path(models_dir)
    if set(selected) != set(TASKS) and not reuse_existing and not overwrite_models:
        existing = [
            str(artifact_paths(output_directory, task, architecture)[0])
            for task in selected
            if artifact_paths(output_directory, task, architecture)[0].is_file()
        ]
        if existing:
            raise FileExistsError(
                f"Model sudah ada: {existing}. Pakai --models-dir lain agar model lama aman, "
                "atau tambahkan --overwrite-models untuk menimpanya."
            )

    run: RunLog | None = None
    if logs_dir is not None:
        try:
            device = describe_device()
        except Exception:  # noqa: BLE001 - metadata perangkat bersifat best-effort
            device = "unknown"
        run = start_run(
            Path(logs_dir),
            run_name,
            architecture,
            {
                "tasks": selected,
                "architecture": architecture,
                "seed": seed,
                "batch_size": batch_size,
                "head_epochs": head_epochs,
                "fine_tune_epochs": fine_tune_epochs,
                "head_learning_rate": HEAD_LEARNING_RATE,
                "fine_tune_learning_rate": FINE_TUNE_LEARNING_RATE,
                "fine_tune_layers": FINE_TUNE_LAYERS,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "input_size": list(INPUT_SIZE),
                "data_root": str(root),
                "models_dir": str(output_directory),
                "defect_data_dir": str(defect_data_dir) if defect_data_dir else None,
                "split_manifest": str(split_manifest) if split_manifest else None,
                "reuse_existing": reuse_existing,
                "primary_crop": primary_crop,
                "device": device,
            },
        )
    try:
        return _run_training(
            root,
            output_directory,
            selected,
            seed,
            batch_size,
            head_epochs,
            fine_tune_epochs,
            reuse_existing,
            architecture,
            Path(defect_data_dir) if defect_data_dir else None,
            Path(split_manifest) if split_manifest else None,
            run,
            tensorboard,
            primary_crop,
        )
    except BaseException:
        logger.exception("Training berhenti karena error.")
        raise
    finally:
        if run is not None:
            run.close()


def measure_efficiency(
    ripeness_model: Any, defect_model: Any, architecture: str
) -> dict[str, Any]:
    """Parameter count and per-image latency (both heads) on the current runtime."""
    return {
        "parameters_total": count_parameters([ripeness_model, defect_model]),
        "latency_ms": benchmark_latency(
            [ripeness_model, defect_model],
            get_preprocess_input(architecture),
            INPUT_SIZE,
        ),
        "device": describe_device(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--architecture",
        nargs="+",
        choices=sorted(ARCHITECTURES),
        default=[DEFAULT_ARCHITECTURE],
        help="Satu atau lebih arsitektur, dilatih berurutan dengan split dan hyperparameter sama.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument(
        "--reuse-existing-models",
        action="store_true",
        help="Reuse existing .keras checkpoints instead of training those tasks again.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(TASKS),
        default=list(TASKS),
        help="Head yang dilatih (bawaan: keduanya). Contoh: --tasks defect.",
    )
    parser.add_argument(
        "--defect-data-dir",
        type=Path,
        default=None,
        help="Folder dataset cacat versi baru yang berisi subfolder healthy/ dan reject/.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="CSV (file,split,group) untuk split per grup; hanya untuk satu task per run.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Folder induk log run (train.log, history.csv, report.md, grafik).",
    )
    parser.add_argument("--run-name", default="train", help="Awalan nama folder run.")
    parser.add_argument(
        "--tensorboard", action="store_true", help="Tulis juga log TensorBoard ke folder run."
    )
    parser.add_argument(
        "--primary-crop",
        choices=["auto", "none", "neutral", "keep"],
        default="auto",
        help="Masukan evaluasi foto primer model cacat. auto: ikuti cara dataset v2 dibuat "
        "(prepare_report.json), foto penuh bila tidak ada; none: foto penuh; neutral/keep: crop tomat.",
    )
    parser.add_argument(
        "--overwrite-models",
        action="store_true",
        help="Izinkan menimpa model yang sudah ada saat melatih sebagian task.",
    )
    args = parser.parse_args()
    results: dict[str, Any] = {}
    for architecture in args.architecture:
        results[architecture] = train(
            args.data_root,
            args.models_dir,
            args.seed,
            args.batch_size,
            args.head_epochs,
            args.fine_tune_epochs,
            args.reuse_existing_models,
            architecture,
            tasks=args.tasks,
            defect_data_dir=args.defect_data_dir,
            split_manifest=args.split_manifest,
            logs_dir=args.logs_dir,
            run_name=args.run_name,
            tensorboard=args.tensorboard,
            overwrite_models=args.overwrite_models,
            primary_crop=None if args.primary_crop == "none" else args.primary_crop,
        )
        _release_session()
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nLog, tabel metrik (report.md), dan grafik ada di: {args.logs_dir}")


def _release_session() -> None:
    """Free graph/memory between architectures in one process (Colab RAM is limited)."""
    import gc

    import tensorflow as tf

    tf.keras.backend.clear_session()
    gc.collect()


if __name__ == "__main__":
    main()
