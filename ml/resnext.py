"""Keras ResNeXt-101 model and image preprocessing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

INPUT_SIZE = (224, 224)
MODEL_NAME = "ResNeXt101"
RIPENESS_CLASSES = ("matang", "setengah_matang", "mentah")
DEFECT_CLASSES = ("healthy", "reject")
RIPENESS_LEVELS = {"matang": 1, "setengah_matang": 2, "mentah": 3}
DEFECT_LEVELS = {"healthy": 1, "reject": 3}


def _keras_dependencies() -> tuple[Any, Any, Any]:
    try:
        import tensorflow as tf
        from classification_models.tfkeras import Classifiers
    except ImportError as exc:
        raise RuntimeError(
            "Training and inference require TensorFlow and image-classifiers. "
            "Install the dependencies from requirements-ml.txt."
        ) from exc

    model_factory, preprocess_input = Classifiers.get("resnext101")
    return tf, model_factory, preprocess_input


def get_preprocess_input() -> Any:
    """Return the ImageNet preprocessing function for the pretrained backbone."""
    _, _, preprocess_input = _keras_dependencies()
    return preprocess_input


def load_trusted_model(path: str | Path) -> Any:
    """Load a project-generated Keras artifact with its trusted Lambda layers."""
    tf, _, _ = _keras_dependencies()
    return tf.keras.models.load_model(path, safe_mode=False)


def create_model(
    task: str,
    num_classes: int,
    input_size: tuple[int, int] = INPUT_SIZE,
) -> tuple[Any, Any, Any]:
    """Create a transfer-learning model and return it with its backbone and preprocessor."""
    if task not in {"ripeness", "defect"}:
        raise ValueError(f"Unknown task: {task!r}")
    if num_classes < 2:
        raise ValueError("A classifier must have at least two classes.")

    tf, model_factory, preprocess_input = _keras_dependencies()
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
    model = tf.keras.Model(inputs, outputs, name=f"resnext101_{task}")
    return model, backbone, preprocess_input


def preprocess_image(image: Any) -> Any:
    """Apply the ImageNet preprocessing expected by the ResNeXt-101 weights."""
    return get_preprocess_input()(image)
