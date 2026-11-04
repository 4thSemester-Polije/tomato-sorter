"""Keras ResNeXt models (50 and 101), artifact naming, and preprocessing helpers."""

from __future__ import annotations

import os
import posixpath
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

INPUT_SIZE = (224, 224)
ARCHITECTURES = {
    "resnext50": "ResNeXt50",
    "resnext101": "ResNeXt101",
}
DEFAULT_ARCHITECTURE = "resnext101"
TASKS = ("ripeness", "defect")
# Jumlah layer backbone terakhir yang dibuka saat fine-tuning. Sama untuk semua
# arsitektur agar perbandingan adil.
FINE_TUNE_LAYERS = 30
RIPENESS_CLASSES = ("matang", "setengah_matang", "mentah")
DEFECT_CLASSES = ("healthy", "reject")
RIPENESS_LEVELS = {"matang": 1, "setengah_matang": 2, "mentah": 3}
DEFECT_LEVELS = {"healthy": 1, "reject": 3}


def validate_architecture(architecture: str) -> str:
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture: {architecture!r}. Choose one of {sorted(ARCHITECTURES)}."
        )
    return architecture


def display_name(architecture: str) -> str:
    return ARCHITECTURES[validate_architecture(architecture)]


def artifact_paths(models_dir: str | Path, task: str, architecture: str) -> tuple[Path, Path]:
    """Return (.keras path, .json metadata path) for one task and architecture."""
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task!r}")
    stem = f"{task}_{validate_architecture(architecture)}"
    directory = Path(models_dir)
    return directory / f"{stem}.keras", directory / f"{stem}.json"


def report_path(models_dir: str | Path, architecture: str) -> Path:
    return Path(models_dir) / f"evaluation_{validate_architecture(architecture)}.json"


def _keras_dependencies(architecture: str) -> tuple[Any, Any, Any]:
    validate_architecture(architecture)
    try:
        import tensorflow as tf
        from classification_models.tfkeras import Classifiers
    except ImportError as exc:
        raise RuntimeError(
            "Training and inference require TensorFlow and image-classifiers. "
            "Install the dependencies from requirements-ml.txt."
        ) from exc

    model_factory, preprocess_input = Classifiers.get(architecture)
    return tf, model_factory, preprocess_input


def get_preprocess_input(architecture: str = DEFAULT_ARCHITECTURE) -> Any:
    """Return the ImageNet preprocessing function for the pretrained backbone."""
    _, _, preprocess_input = _keras_dependencies(architecture)
    return preprocess_input


@contextmanager
def posix_keras_paths(tf: Any, enabled: bool | None = None) -> Iterator[None]:
    """Make Keras 2.15 weight-store paths use '/' on Windows while a model loads.

    Keras 2.15 builds the HDF5 group paths inside a ``.keras`` archive with
    ``tf.io.gfile.join``, which is ``os.path.join`` and therefore uses ``\\`` on
    Windows. A ``.keras`` file written on Linux (Google Colab) stores them with
    ``/``, so on Windows every nested layer (the ResNeXt backbone) is looked up
    under a name that does not exist and loads as empty. The error is
    ``Layer 'bn_data' expected 3 variables, but received 0 variables``.
    Absolute paths (temporary directories) keep the native behaviour.
    """
    if enabled is None:
        enabled = os.name == "nt"
    if not enabled:
        yield
        return

    modules = []
    for candidate in (tf.io.gfile, tf.compat.v2.io.gfile):
        if all(candidate is not seen for seen, _ in modules):
            modules.append((candidate, candidate.join))

    def make_join(original: Any) -> Any:
        def join(path: Any, *paths: Any) -> Any:
            text = os.fspath(path) if not isinstance(path, (str, bytes)) else path
            if isinstance(text, bytes) or "://" in text[1:] or os.path.isabs(text):
                return original(path, *paths)
            return posixpath.join(text, *paths)

        return join

    for module, original in modules:
        module.join = make_join(original)
    try:
        yield
    finally:
        for module, original in modules:
            module.join = original


def load_trusted_model(path: str | Path) -> Any:
    """Load a project-generated Keras artifact with its trusted Lambda layers."""
    # Memuat dependensi lengkap (termasuk image-classifiers) seperti sebelumnya,
    # karena backbone ResNeXt berisi layer Lambda dari pustaka tersebut.
    tf, _, _ = _keras_dependencies(DEFAULT_ARCHITECTURE)
    with posix_keras_paths(tf):
        return tf.keras.models.load_model(path, safe_mode=False)


def create_model(
    task: str,
    num_classes: int,
    input_size: tuple[int, int] = INPUT_SIZE,
    *,
    architecture: str = DEFAULT_ARCHITECTURE,
) -> tuple[Any, Any, Any]:
    """Create a transfer-learning model and return it with its backbone and preprocessor."""
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task!r}")
    if num_classes < 2:
        raise ValueError("A classifier must have at least two classes.")

    tf, model_factory, preprocess_input = _keras_dependencies(architecture)
    backbone = model_factory(
        input_shape=(*input_size, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(*input_size, 3), name="image")
    augmented = tf.keras.layers.RandomFlip(
        mode="horizontal_and_vertical", name="random_flip"
    )(inputs)
    augmented = tf.keras.layers.RandomRotation(
        factor=0.08, name="random_rotation"
    )(augmented)
    features = backbone(augmented, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D(name="average_pool")(features)
    features = tf.keras.layers.Dropout(0.3, name="dropout")(features)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="probabilities"
    )(features)
    model = tf.keras.Model(inputs, outputs, name=f"{architecture}_{task}")
    return model, backbone, preprocess_input


def preprocess_image(image: Any, architecture: str = DEFAULT_ARCHITECTURE) -> Any:
    """Apply the ImageNet preprocessing expected by the ResNeXt weights."""
    return get_preprocess_input(architecture)(image)
