from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    roc_auc_score,
    roc_curve,
)

from ml.resnext import (
    DEFAULT_ARCHITECTURE,
    INPUT_SIZE,
    RIPENESS_CLASSES,
    DEFECT_CLASSES,
    artifact_paths,
    get_preprocess_input,
    load_trusted_model,
)
from ml.train import _primary_test_samples


def predict(model, image, preprocess):
    """Prediksi satu gambar PIL dengan preprocessing yang sama seperti API."""
    resized = image.convert("RGB").resize(
        (INPUT_SIZE[1], INPUT_SIZE[0])
    )
    pixels = np.asarray(resized, dtype=np.float32)
    batch = preprocess(np.expand_dims(pixels, axis=0))
    return model.predict(batch, verbose=0)[0]


def crop_tomato(image):
    """
    Crop berdasarkan segmentasi vision yang ada di repository.
    Mengembalikan (gambar_crop, status).
    """
    rgb = np.asarray(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    from vision.segmentation import segment

    # Tanpa background: gunakan segmentasi saturasi.
    mask = segment(bgr)

    points = cv2.findNonZero(mask)
    if points is None:
        return None, "mask_kosong"

    x, y, w, h = cv2.boundingRect(points)

    # Tambahkan padding 8% agar bagian tepi tomat tidak terpotong.
    pad_x = int(w * 0.08)
    pad_y = int(h * 0.08)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(bgr.shape[1], x + w + pad_x)
    y2 = min(bgr.shape[0], y + h + pad_y)

    cropped = bgr[y1:y2, x1:x2]
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    return Image.fromarray(cropped_rgb), "ok"


def save_roc_plot(y_true, probabilities, class_names, output_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))

    for index, class_name in enumerate(class_names):
        binary_true = (np.asarray(y_true) == index).astype(int)

        if len(np.unique(binary_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(
            binary_true, probabilities[:, index]
        )
        class_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{class_name} (AUC={class_auc:.3f})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--models-dir", default=Path("models"), type=Path)
    parser.add_argument(
        "--architecture",
        default=DEFAULT_ARCHITECTURE,
        choices=["resnext50", "resnext101"],
    )
    parser.add_argument(
        "--output-dir",
        default=Path("reports/diagnosis"),
        type=Path,
    )
    args = parser.parse_args()

    output = args.output_dir
    gallery = output / "false_positive"
    output.mkdir(parents=True, exist_ok=True)
    gallery.mkdir(parents=True, exist_ok=True)

    # Muat model dan metadata yang sudah dilatih.
    ripeness_path, ripeness_meta_path = artifact_paths(
        args.models_dir, "ripeness", args.architecture
    )
    defect_path, defect_meta_path = artifact_paths(
        args.models_dir, "defect", args.architecture
    )

    for path in (
        ripeness_path,
        ripeness_meta_path,
        defect_path,
        defect_meta_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

    ripeness_model = load_trusted_model(ripeness_path)
    defect_model = load_trusted_model(defect_path)

    ripeness_meta = json.loads(
        ripeness_meta_path.read_text(encoding="utf-8")
    )
    defect_meta = json.loads(
        defect_meta_path.read_text(encoding="utf-8")
    )

    ripeness_classes = ripeness_meta["class_names"]
    defect_classes = defect_meta["class_names"]
    preprocess = get_preprocess_input(args.architecture)

    samples = _primary_test_samples(args.data_root)

    rows = []
    ripeness_true, ripeness_probs = [], []
    defect_true, defect_probs = [], []

    crop_correct = 0
    crop_total = 0
    full_correct = 0
    full_total = 0

    false_positive_count = 0

    print(f"Jumlah foto primer: {len(samples)}")

    for index, (path, labels) in enumerate(samples, start=1):
        with Image.open(path) as source:
            image = source.convert("RGB")

        # Prediksi foto utuh.
        rp = predict(ripeness_model, image, preprocess)
        dp = predict(defect_model, image, preprocess)

        actual_ripeness = int(labels["tingkat_kematangan"]) - 1
        actual_defect_level = int(labels["tingkat_cacat"])

        pred_ripeness_index = int(np.argmax(rp))
        pred_defect_index = int(np.argmax(dp))

        pred_ripeness = ripeness_classes[pred_ripeness_index]
        pred_defect = defect_classes[pred_defect_index]

        full_correct += int(pred_ripeness_index == actual_ripeness)
        full_total += 1

        ripeness_true.append(actual_ripeness)
        ripeness_probs.append(rp)

        # Level 2 tidak dimasukkan ke evaluasi AUC biner.
        if actual_defect_level in (1, 3):
            actual_defect = 0 if actual_defect_level == 1 else 1
            defect_true.append(actual_defect)
            defect_probs.append(dp)

            # False positive: label asli healthy, prediksi reject.
            if actual_defect == 0 and pred_defect == "reject":
                false_positive_count += 1
                shutil.copy2(
                    path,
                    gallery / f"{false_positive_count:03d}_{path.name}",
                )

        # Uji crop.
        cropped, crop_status = crop_tomato(image)
        crop_pred = None

        if cropped is not None:
            crop_rp = predict(
                ripeness_model, cropped, preprocess
            )
            crop_index = int(np.argmax(crop_rp))
            crop_pred = ripeness_classes[crop_index]

            crop_correct += int(crop_index == actual_ripeness)
            crop_total += 1

        rows.append({
            "filename": path.name,
            "actual_ripeness": ripeness_classes[actual_ripeness],
            "predicted_ripeness": pred_ripeness,
            "ripeness_confidence": float(rp[pred_ripeness_index]),
            "actual_defect_level": actual_defect_level,
            "predicted_defect": pred_defect,
            "defect_confidence": float(dp[pred_defect_index]),
            "prob_healthy": float(dp[defect_classes.index("healthy")]),
            "prob_reject": float(dp[defect_classes.index("reject")]),
            "crop_status": crop_status,
            "crop_predicted_ripeness": crop_pred,
        })

        print(f"[{index}/{len(samples)}] {path.name}")

    # Simpan prediksi per gambar.
    csv_path = output / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # AUC kematangan: multiclass One-vs-Rest.
    y_r = np.asarray(ripeness_true)
    p_r = np.asarray(ripeness_probs)

    ripeness_auc = roc_auc_score(
        y_r,
        p_r,
        multi_class="ovr",
        average="macro",
        labels=list(range(len(ripeness_classes))),
    )

    save_roc_plot(
        y_r,
        p_r,
        ripeness_classes,
        output / "roc_ripeness.png",
    )

    # AUC cacat: binary Healthy vs Reject.
    y_d = np.asarray(defect_true)
    p_d = np.asarray(defect_probs)
    reject_index = defect_classes.index("reject")

    defect_auc = roc_auc_score(
        y_d,
        p_d[:, reject_index],
    )

    fpr, tpr, _ = roc_curve(y_d, p_d[:, reject_index])

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr,
        tpr,
        label=f"Reject (AUC={defect_auc:.3f})",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Defect")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / "roc_defect.png", dpi=150)
    plt.close()

    report = {
        "architecture": args.architecture,
        "total_primary_images": len(samples),
        "ripeness": {
            "auc_macro_ovr": float(ripeness_auc),
            "accuracy_full_frame": float(
                accuracy_score(
                    y_r,
                    np.argmax(p_r, axis=1),
                )
            ),
            "class_names": ripeness_classes,
        },
        "defect": {
            "auc": float(defect_auc),
            "evaluated_images": len(y_d),
            "excluded_level_2_images": sum(
                int(row["actual_defect_level"] == 2)
                for row in rows
            ),
            "false_positive_healthy_as_reject": false_positive_count,
        },
        "cropping": {
            "valid_crop_images": crop_total,
            "accuracy_full_frame": (
                full_correct / full_total if full_total else None
            ),
            "accuracy_crop": (
                crop_correct / crop_total if crop_total else None
            ),
        },
    }

    (output / "auc_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== HASIL DIAGNOSIS ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nHasil tersimpan di: {output.resolve()}")


if __name__ == "__main__":
    main()
