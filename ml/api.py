"""HTTP API for the trained ResNeXt (50 / 101) tomato classifiers."""

from __future__ import annotations

import io
import json
import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
import pillow_heif

from ml.resnext import (
    ARCHITECTURES,
    DEFAULT_ARCHITECTURE,
    DEFECT_LEVELS,
    INPUT_SIZE,
    RIPENESS_LEVELS,
    TASKS,
    artifact_paths,
    display_name,
    get_preprocess_input,
    load_trusted_model,
    validate_architecture,
)
from ml.rubric import grade_from_levels

MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_MODELS_DIR = Path(os.environ.get("TOMATO_MODELS_DIR", "models"))
ARCHITECTURE_PATTERN = "^(" + "|".join(sorted(ARCHITECTURES)) + ")$"
pillow_heif.register_heif_opener()
logger = logging.getLogger("uvicorn.error")


def warm_up(models_dir: Path | None = None, architecture: str | None = None) -> bool:
    """Muat model dan jalankan satu prediksi kosong agar tomat pertama tidak menunggu pemuatan model.

    Tanpa ini, permintaan nyata pertama memuat model (puluhan detik di CPU) dan bisa melewati batas
    firmware menunggu GRADE (ERR GRADE_TIMEOUT). Kegagalan tidak menghentikan server: /health tetap
    melaporkan model yang belum siap.
    """
    directory = DEFAULT_MODELS_DIR if models_dir is None else models_dir
    try:
        chosen = architecture or _default_architecture()
        predict_image(Image.new("RGB", INPUT_SIZE, (128, 128, 128)), directory, chosen)
    except Exception as exc:  # noqa: BLE001 - server harus tetap hidup agar /health bisa menjelaskan masalahnya
        logger.warning("Warm-up model dilewati: %s", exc)
        return False
    logger.info("Warm-up model selesai (%s).", chosen)
    return True


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.environ.get("TOMATO_WARMUP", "1") != "0":
        warm_up()  # server baru menerima permintaan setelah ini selesai
    yield


app = FastAPI(
    title="Tomato Sorter ML API",
    version="1.2.0",
    description="ResNeXt-50/101 predictions for tomato ripeness, visible defects, and grade.",
    lifespan=lifespan,
)


def _default_architecture() -> str:
    """Architecture used when a request does not pick one (env TOMATO_ARCHITECTURE)."""
    return validate_architecture(os.environ.get("TOMATO_ARCHITECTURE", DEFAULT_ARCHITECTURE))


def _required_files(models_dir: Path, architecture: str) -> list[Path]:
    paths: list[Path] = []
    for task in TASKS:
        paths.extend(artifact_paths(models_dir, task, architecture))
    return paths


@lru_cache(maxsize=4)
def _load_models(
    models_dir: str, architecture: str = DEFAULT_ARCHITECTURE
) -> tuple[Any, Any, dict[str, Any], dict[str, Any], Any]:
    directory = Path(models_dir)
    ripeness_files = artifact_paths(directory, "ripeness", architecture)
    defect_files = artifact_paths(directory, "defect", architecture)
    paths = (ripeness_files[0], defect_files[0], ripeness_files[1], defect_files[1])
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Trained model files are missing: " + ", ".join(missing)
        )
    ripeness_model = load_trusted_model(paths[0])
    defect_model = load_trusted_model(paths[1])
    ripeness_metadata = json.loads(
        paths[2].read_text(encoding="utf-8")
    )
    defect_metadata = json.loads(
        paths[3].read_text(encoding="utf-8")
    )
    preprocess_input = get_preprocess_input(architecture)
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


def predict_image(
    image: Image.Image,
    models_dir: Path = DEFAULT_MODELS_DIR,
    architecture: str = DEFAULT_ARCHITECTURE,
) -> dict[str, Any]:
    """Run both classifiers and combine their levels using the project rubric."""
    (
        ripeness_model,
        defect_model,
        ripeness_metadata,
        defect_metadata,
        preprocess_input,
    ) = _load_models(str(models_dir.resolve()), architecture)
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
        "model": display_name(architecture),
        "architecture": architecture,
        "input_size": list(INPUT_SIZE),
        "ripeness": ripeness,
        "defect": defect,
        "grade": grade_from_levels(ripeness["level"], defect["level"]),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    default = _default_architecture()
    architectures: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        missing = [
            path.name
            for path in _required_files(DEFAULT_MODELS_DIR, architecture)
            if not path.is_file()
        ]
        architectures[architecture] = {
            "status": "ready" if not missing else "model_missing",
            "missing_files": missing,
        }
    return {
        "status": architectures[default]["status"],
        "model": display_name(default),
        "architecture": default,
        "missing_files": architectures[default]["missing_files"],
        "architectures": architectures,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    architecture: str | None = Query(
        default=None,
        pattern=ARCHITECTURE_PATTERN,
        description="resnext50 atau resnext101. Kosong = TOMATO_ARCHITECTURE (bawaan resnext101).",
    ),
) -> dict[str, Any]:
    chosen = architecture or _default_architecture()
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
        return predict_image(decoded_image, DEFAULT_MODELS_DIR, chosen)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
