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

GRADE_USES_DEFECT_ENV = "TOMATO_GRADE_USES_DEFECT"
# Ditampilkan ke controller/dashboard bila kepala cacat tidak ikut menentukan grade.
NOT_ASSESSED_DEFECT: dict[str, Any] = {
    "class": "tidak_dinilai",
    "level": None,
    "confidence": 0.0,
    "probabilities": {},
}


DEFECT_THRESHOLD_ENV = "TOMATO_DEFECT_THRESHOLD"
DEFAULT_DEFECT_THRESHOLD = 0.5


def _defect_threshold() -> float:
    """Ambang p_reject untuk menyatakan cacat jelas (bawaan 0,5). Atur lewat TOMATO_DEFECT_THRESHOLD."""
    raw = os.environ.get(DEFECT_THRESHOLD_ENV)
    if raw is None or raw.strip() == "":
        return DEFAULT_DEFECT_THRESHOLD
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")
    if not 0.0 < value < 1.0:
        logger.warning(
            "%s=%r tidak valid (harus di antara 0 dan 1); memakai %.2f.",
            DEFECT_THRESHOLD_ENV, raw, DEFAULT_DEFECT_THRESHOLD,
        )
        return DEFAULT_DEFECT_THRESHOLD
    return value


def _not_assessed(reason: str) -> dict[str, Any]:
    return {**NOT_ASSESSED_DEFECT, "probabilities": {}, "reason": reason}


def _defect_training_input(metadata: dict[str, Any]) -> str | None:
    """"neutral"/"keep" bila model cacat dilatih pada crop tomat (metadata dari ml.train + prepare_defect_v2).

    None berarti model lama yang dilatih pada foto penuh.
    """
    background = (metadata.get("data_preparation") or {}).get("background")
    return background if background in ("neutral", "keep") else None


def _predict_defect(
    defect_model: Any,
    defect_metadata: dict[str, Any],
    preprocess_input: Any,
    processed: np.ndarray,
    defect_source: Image.Image | None,
    threshold: float,
) -> dict[str, Any]:
    """Model lama: argmax pada masukan yang sama dengan kematangan. Model crop: crop dari frame mentah."""
    mode = _defect_training_input(defect_metadata)
    if mode is None:
        return _class_prediction(
            defect_model.predict(processed, verbose=0)[0], defect_metadata["class_names"], DEFECT_LEVELS
        )
    if defect_source is None:
        # Model dilatih pada crop mentah; gambar PCV (CLAHE) bukan masukan yang sama. Jangan menebak.
        return _not_assessed("tanpa_frame_mentah")
    from vision.crop import crop_for_inference

    crop = crop_for_inference(defect_source, mode)
    if crop is None:
        return _not_assessed("crop_gagal")
    pixels = np.asarray(crop.resize((INPUT_SIZE[1], INPUT_SIZE[0])), dtype=np.float32)
    probabilities = defect_model.predict(preprocess_input(np.expand_dims(pixels, axis=0)), verbose=0)[0]
    names = list(defect_metadata["class_names"])
    p_reject = float(probabilities[names.index("reject")])
    chosen = "reject" if p_reject >= threshold else "healthy"
    return {
        "class": chosen,
        "level": DEFECT_LEVELS[chosen],
        "confidence": p_reject if chosen == "reject" else 1.0 - p_reject,
        "probabilities": {label: float(probabilities[i]) for i, label in enumerate(names)},
        "p_reject": p_reject,
        "threshold": threshold,
        "input": f"crop_{mode}",
    }


def _grade_uses_defect() -> bool:
    """Bawaan: grade hanya dari kematangan. Atur TOMATO_GRADE_USES_DEFECT=1 agar kepala cacat ikut dipakai.

    Alasan: pada 108 foto primer, kepala cacat (dilatih dari dataset Kaggle yang definisi "Reject"-nya
    berbeda dari rubrik) menandai ~50% tomat normal sebagai reject, sehingga hampir semua jadi grade C.
    """
    return os.environ.get(GRADE_USES_DEFECT_ENV, "0") == "1"


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    """Bentuk respons untuk controller/dashboard. predict_image() tetap mengembalikan keluaran CNN apa adanya
    (dipakai alat evaluasi); di sini tebakan cacat yang tidak ikut grade tidak ditampilkan sebagai hasil."""
    if result.get("grade_basis") != "ripeness":
        return result
    shown = dict(result)
    raw = result.get("defect") or {}
    shown["defect_raw"] = raw
    shown["defect"] = {
        **NOT_ASSESSED_DEFECT,
        "reason": raw.get("reason") if raw.get("level") is None else "tidak_ikut_grade",
    }
    return shown


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
        _warm_defect_head(directory, chosen)
    except Exception as exc:  # noqa: BLE001 - server harus tetap hidup agar /health bisa menjelaskan masalahnya
        logger.warning("Warm-up model dilewati: %s", exc)
        return False
    logger.info("Warm-up model selesai (%s).", chosen)
    return True


def _warm_defect_head(directory: Path, architecture: str) -> None:
    """Panaskan jalur crop + model cacat bila model cacat dilatih pada crop (best-effort, tidak menggagalkan warm-up)."""
    try:
        _, _, _, defect_metadata, _ = _load_models(str(directory.resolve()), architecture)
        if _defect_training_input(defect_metadata) is None:
            return
        predict_image(
            Image.new("RGB", INPUT_SIZE, (128, 128, 128)),
            directory,
            architecture,
            defect_source=_warmup_scene(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Pemanasan model cacat dilewati: %s", exc)


def _warmup_scene() -> Image.Image:
    """Tomat sintetis agar jalur crop dan model cacat ikut dipanaskan (bukan hanya model kematangan)."""
    from PIL import ImageDraw

    scene = Image.new("RGB", (640, 480), (255, 255, 255))
    ImageDraw.Draw(scene).ellipse((220, 140, 420, 340), fill=(200, 40, 30))
    return scene


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
    defect_source: Image.Image | None = None,
) -> dict[str, Any]:
    """Run both classifiers and combine their levels using the project rubric.

    defect_source: frame mentah untuk model cacat yang dilatih pada crop tomat (lihat
    _defect_training_input). Diabaikan oleh model cacat lama (foto penuh).
    """
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

    ripeness = _class_prediction(
        ripeness_probabilities,
        ripeness_metadata["class_names"],
        RIPENESS_LEVELS,
    )
    defect = _predict_defect(
        defect_model,
        defect_metadata,
        preprocess_input,
        processed,
        defect_source,
        _defect_threshold(),
    )
    # Cacat ikut grade hanya bila diaktifkan DAN berhasil dinilai (crop gagal tidak boleh dianggap sehat atau cacat).
    uses_defect = _grade_uses_defect() and defect["level"] is not None
    return {
        "model": display_name(architecture),
        "architecture": architecture,
        "input_size": list(INPUT_SIZE),
        "ripeness": ripeness,
        "defect": defect,
        "grade": grade_from_levels(ripeness["level"], defect["level"] if uses_defect else 1),
        "grade_basis": "ripeness+defect" if uses_defect else "ripeness",
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
        "defect_head": _defect_head_info(DEFAULT_MODELS_DIR, default),
    }


def _defect_head_info(models_dir: Path, architecture: str) -> dict[str, Any]:
    """Model cacat mana yang dipakai dan bagaimana: berkas, masukan saat training, ambang, ikut grade atau tidak."""
    model_path, metadata_path = artifact_paths(models_dir, "defect", architecture)
    info: dict[str, Any] = {
        "model_file": str(model_path),
        "model_exists": model_path.is_file(),
        "trained_input": None,
        "threshold": _defect_threshold(),
        "used_for_grade": _grade_uses_defect(),
    }
    if model_path.is_file():
        info["model_size_mb"] = round(model_path.stat().st_size / 1e6, 1)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return info
    mode = _defect_training_input(metadata)
    info["trained_input"] = f"crop_{mode}" if mode else "full_frame"
    info["data_preparation"] = metadata.get("data_preparation")
    return info


async def _read_image(upload: UploadFile) -> Image.Image:
    content = await upload.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MiB limit.")
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=422,
            detail="Uploaded file is not a readable JPG, PNG, or supported HEIC image.",
        ) from exc


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    frame: UploadFile | None = File(
        default=None,
        description="Frame mentah (opsional). Dipakai model cacat yang dilatih pada crop tomat; "
        "tanpa ini, model cacat seperti itu melaporkan tidak_dinilai.",
    ),
    architecture: str | None = Query(
        default=None,
        pattern=ARCHITECTURE_PATTERN,
        description="resnext50 atau resnext101. Kosong = TOMATO_ARCHITECTURE (bawaan resnext101).",
    ),
    apply_pcv: bool = Query(
        default=False,
        description="Jika true, jalankan pipeline PCV (Crop 1:1 + HSV + Tekstur CLAHE/LAB) sebelum prediksi ResNeXt.",
    ),
) -> dict[str, Any]:
    chosen = architecture or _default_architecture()
    decoded_image = await _read_image(file)
    frame_image = await _read_image(frame) if frame is not None else None
    raw_image = decoded_image

    pcv_summary: dict[str, Any] | None = None
    if apply_pcv:
        import cv2
        from vision.features import extract_pcv_features

        frame_bgr = cv2.cvtColor(np.asarray(decoded_image), cv2.COLOR_RGB2BGR)
        pcv = extract_pcv_features(frame_bgr, crop_size=300)
        decoded_image = Image.fromarray(pcv["img_enhanced_rgb"])
        pcv_summary = {
            "persen_merah": round(pcv["persen_merah"], 2),
            "persen_oranye": round(pcv["persen_oranye"], 2),
            "persen_kuning": round(pcv["persen_kuning"], 2),
            "persen_hijau": round(pcv["persen_hijau"], 2),
            "persentase_cacat": round(pcv["persentase_cacat"], 2),
            "jumlah_kontur_cacat": pcv["jumlah_kontur_cacat"],
            "luas_piksel": round(pcv["luas_piksel"], 1),
        }

    # Dengan apply_pcv gambar asli (sebelum PCV) otomatis menjadi sumber model cacat.
    defect_source = frame_image if frame_image is not None else (raw_image if apply_pcv else None)
    options = {"defect_source": defect_source} if defect_source is not None else {}
    try:
        result = _public_result(predict_image(decoded_image, DEFAULT_MODELS_DIR, chosen, **options))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if pcv_summary is not None:
        result = {**result, "pcv": pcv_summary}
    return result
