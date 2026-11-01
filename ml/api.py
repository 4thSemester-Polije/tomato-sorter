"""HTTP API for the trained ResNeXt-101 tomato classifiers."""

from __future__ import annotations

import io
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
import pillow_heif

from ml.resnext import (
    DEFECT_LEVELS,
    INPUT_SIZE,
    MODEL_NAME,
    RIPENESS_LEVELS,
    get_preprocess_input,
)
from ml.rubric import grade_from_levels

MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_MODELS_DIR = Path(os.environ.get("TOMATO_MODELS_DIR", "models"))
pillow_heif.register_heif_opener()

app = FastAPI(
    title="Tomato Sorter ML API",
    version="1.0.0",
    description="ResNeXt-101 predictions for tomato ripeness, visible defects, and grade.",
)


@lru_cache(maxsize=4)
def _load_models(models_dir: str) -> tuple[Any, Any, dict[str, Any], dict[str, Any], Any]:
    directory = Path(models_dir)
    paths = (
        directory / "ripeness_resnext101.keras",
        directory / "defect_resnext101.keras",
        directory / "ripeness_resnext101.json",
        directory / "defect_resnext101.json",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Trained model files are missing: " + ", ".join(missing)
        )
    import tensorflow as tf

    ripeness_model = tf.keras.models.load_model(paths[0])
    defect_model = tf.keras.models.load_model(paths[1])
    ripeness_metadata = json.loads(
        paths[2].read_text(encoding="utf-8")
    )
    defect_metadata = json.loads(
        paths[3].read_text(encoding="utf-8")
    )
    preprocess_input = get_preprocess_input()
    return (
        ripeness_model,
        defect_model,
        ripeness_metadata,
        defect_metadata,
        preprocess_input,
    )


def _class_prediction(
    probabilities: np.ndarray,
    class_names: list[str],
    class_levels: dict[str, int],
) -> dict[str, Any]:
    index = int(np.argmax(probabilities))
    name = class_names[index]
    return {
        "class": name,
        "level": class_levels[name],
        "confidence": float(probabilities[index]),
        "probabilities": {
            label: float(probabilities[position])
            for position, label in enumerate(class_names)
        },
    }


def predict_image(image: Image.Image, models_dir: Path = DEFAULT_MODELS_DIR) -> dict[str, Any]:
    """Run both classifiers and combine their levels using the project rubric."""
    (
        ripeness_model,
        defect_model,
        ripeness_metadata,
        defect_metadata,
        preprocess_input,
    ) = _load_models(str(models_dir.resolve()))
    resized = image.convert("RGB").resize((INPUT_SIZE[1], INPUT_SIZE[0]))
    pixels = np.asarray(resized, dtype=np.float32)
    processed = preprocess_input(np.expand_dims(pixels, axis=0))
    ripeness_probabilities = ripeness_model.predict(processed, verbose=0)[0]
    defect_probabilities = defect_model.predict(processed, verbose=0)[0]

    ripeness = _class_prediction(
        ripeness_probabilities,
        ripeness_metadata["class_names"],
        RIPENESS_LEVELS,
    )
    defect = _class_prediction(
        defect_probabilities,
        defect_metadata["class_names"],
        DEFECT_LEVELS,
    )
    return {
        "model": MODEL_NAME,
        "input_size": list(INPUT_SIZE),
        "ripeness": ripeness,
        "defect": defect,
        "grade": grade_from_levels(ripeness["level"], defect["level"]),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    required = (
        DEFAULT_MODELS_DIR / "ripeness_resnext101.keras",
        DEFAULT_MODELS_DIR / "defect_resnext101.keras",
        DEFAULT_MODELS_DIR / "ripeness_resnext101.json",
        DEFAULT_MODELS_DIR / "defect_resnext101.json",
    )
    missing = [path.name for path in required if not path.is_file()]
    return {
        "status": "ready" if not missing else "model_missing",
        "model": MODEL_NAME,
        "missing_files": missing,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    content = await file.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MiB limit.")
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            decoded_image = image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=422,
            detail="Uploaded file is not a readable JPG, PNG, or supported HEIC image.",
        ) from exc

    try:
        return predict_image(decoded_image, DEFAULT_MODELS_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
