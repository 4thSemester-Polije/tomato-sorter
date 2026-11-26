"""Siapkan dataset cacat v2 dari AgML (Fresh/Rotten): crop per tomat, kelompok split, manifest.

Keluaran di --output-dir:
    healthy/            crop tomat sehat (dari Fresh)
    reject/             crop tomat cacat jelas (dari Rotten)
    split_manifest.csv  file,split,group  (dipakai ml.train --split-manifest)
    crops_index.csv     metadata tiap crop (sumber, kotak, solidity, ...)
    prepare_report.md/.json  ringkasan, alasan pengecualian, statistik kelompok
    qa/                 contact sheet untuk pemeriksaan visual

Perlindungan terhadap kebocoran dan jalan pintas:
- Crop persegi dengan latar netral (opsional) agar meja/kertas/sabuk tidak jadi petunjuk.
- Gambar dengan tomat bersentuhan/menyatu ditolak; foto Rotten bermultitomat dikecualikan karena
  label gambarnya belum tentu berlaku untuk setiap buah.
- Satu kelompok = crop dari foto yang sama, foto bernomor berdekatan (satu sesi pemotretan), dan
  crop yang hampir identik. Semua anggota kelompok masuk split yang sama.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

from ml.run_logging import render_table
from vision.crop import (
    MAX_SIDE,
    MIN_SOLIDITY,
    crop_region,
    find_tomato_regions,
    is_clean_single_tomato,
    pil_to_bgr,
)

logger = logging.getLogger(__name__)

SOURCE_FOLDERS = {"fresh": "healthy", "rotten": "reject"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
THUMBNAIL_SIZE = 24
MANAGED_ENTRIES = (
    "healthy",
    "reject",
    "qa",
    "split_manifest.csv",
    "crops_index.csv",
    "prepare_report.md",
    "prepare_report.json",
)
REASONS = {
    "tidak_ada_tomat": "tidak ada objek tomat terdeteksi",
    "tomat_menyatu_atau_terpotong": "tomat menyentuh tepi gambar atau bentuknya tidak cembung",
    "multi_tomat_rotten": "foto Rotten dengan beberapa tomat (label per buah tidak pasti)",
    "gagal_dibaca": "file gambar tidak bisa dibaca",
    "label_konflik": "gambar nyaris identik muncul di Fresh dan Rotten",
}


@dataclass
class CropRecord:
    file: str  # "<label>/<nama file>"
    label: str
    source: str
    index: int | None
    box: tuple[int, int, int, int]
    area_fraction: float
    solidity: float
    regions_in_source: int
    thumbnail: np.ndarray


# --------------------------------------------------------------------------- input


def load_bgr(path: Path, max_side: int = MAX_SIDE) -> np.ndarray:
    with Image.open(path) as source:
        return pil_to_bgr(ImageOps.exif_transpose(source), max_side)


def list_sources(agml_dir: Path, limit: int | None = None) -> list[tuple[Path, str]]:
    folders = {p.name.lower(): p for p in Path(agml_dir).iterdir() if p.is_dir()} if Path(agml_dir).is_dir() else {}
    missing = [name for name in SOURCE_FOLDERS if name not in folders]
    if missing:
        raise FileNotFoundError(
            f"Folder {missing} tidak ditemukan di {agml_dir} (diharapkan Fresh dan Rotten)."
        )
    sources: list[tuple[Path, str]] = []
    for name, label in SOURCE_FOLDERS.items():
        files = sorted(
            p for p in folders[name].iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        sources += [(path, label) for path in (files[:limit] if limit else files)]
    if not sources:
        raise ValueError(f"Tidak ada gambar JPG/PNG di {agml_dir}.")
    return sources


def source_index(stem: str) -> int | None:
    match = re.search(r"(\d+)$", stem)
    return int(match.group(1)) if match else None


def _thumbnail(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    vector = cv2.resize(gray, (THUMBNAIL_SIZE, THUMBNAIL_SIZE), interpolation=cv2.INTER_AREA)
    vector = vector.astype(np.float32).ravel()
    vector -= vector.mean()
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _write_jpeg(path: Path, image_bgr: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"Gagal meng-encode {path}")
    path.write_bytes(encoded.tobytes())


# --------------------------------------------------------------------------- crop


def process_sources(
    sources: Sequence[tuple[Path, str]],
    output_dir: Path,
    size: int = 256,
    neutral_background: bool = True,
    multi_rotten: str = "exclude",
    min_solidity: float = MIN_SOLIDITY,
) -> tuple[list[CropRecord], dict[str, list[tuple[Path, str]]], Counter]:
    records: list[CropRecord] = []
    excluded: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    source_counts: Counter = Counter()
    for label in set(label for _, label in sources):
        (output_dir / label).mkdir(parents=True, exist_ok=True)

    for number, (path, label) in enumerate(sources, start=1):
        source_counts[label] += 1
        if number % 100 == 0:
            logger.info("Diproses %d/%d gambar sumber ...", number, len(sources))
        try:
            image = load_bgr(path)
        except Exception:  # noqa: BLE001 - satu file rusak tidak boleh menghentikan seluruh proses
            logger.exception("Tidak bisa membaca %s", path)
            excluded["gagal_dibaca"].append((path, label))
            continue
        regions = find_tomato_regions(image)
        if not regions:
            excluded["tidak_ada_tomat"].append((path, label))
            continue
        clean = [region for region in regions if is_clean_single_tomato(region, min_solidity)]
        if label == "reject" and multi_rotten == "exclude":
            suspect_multi = len(regions) >= 2 or any(r.solidity < min_solidity for r in regions)
            if suspect_multi:
                excluded["multi_tomat_rotten"].append((path, label))
                continue
        if not clean:
            excluded["tomat_menyatu_atau_terpotong"].append((path, label))
            continue
        for order, region in enumerate(clean, start=1):
            crop = crop_region(image, region, size=size, neutral_background=neutral_background)
            name = f"{path.stem}_t{order}.jpg"
            _write_jpeg(output_dir / label / name, crop)
            records.append(
                CropRecord(
                    file=f"{label}/{name}",
                    label=label,
                    source=path.stem,
                    index=source_index(path.stem),
                    box=region.box,
                    area_fraction=region.area_fraction,
                    solidity=region.solidity,
                    regions_in_source=len(regions),
                    thumbnail=_thumbnail(crop),
                )
            )
    return records, excluded, source_counts


# --------------------------------------------------------------------------- kelompok & split


def assign_groups(
    records: Sequence[CropRecord], group_block: int = 20, dup_threshold: float = 0.985
) -> list[int]:
    """Union-find: foto sama, blok nomor berdekatan, dan crop hampir identik -> satu kelompok."""
    count = len(records)
    parent = list(range(count))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    buckets: dict[Any, list[int]] = defaultdict(list)
    for position, record in enumerate(records):
        buckets[("source", record.label, record.source)].append(position)
        if group_block > 0 and record.index is not None:
            buckets[("block", record.label, record.index // group_block)].append(position)
    for members in buckets.values():
        for other in members[1:]:
            union(members[0], other)

    if dup_threshold > 0 and count > 1:
        matrix = np.stack([record.thumbnail for record in records])
        for start in range(0, count, 500):
            similarity = matrix[start : start + 500] @ matrix.T
            for row, column in np.argwhere(similarity >= dup_threshold):
                if start + int(row) != int(column):
                    union(start + int(row), int(column))

    ids: dict[int, int] = {}
    return [ids.setdefault(find(position), len(ids) + 1) for position in range(count)]


def split_groups(
    records: Sequence[CropRecord],
    groups: Sequence[int],
    seed: int = 42,
    test_fraction: float = 0.15,
    validation_fraction: float = 0.15,
) -> dict[int, str]:
    """Bagi kelompok ke train/validation/test per kelas, mendekati proporsi 70/15/15."""
    members: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for record, group in zip(records, groups):
        members[record.label][group] += 1
    rng = random.Random(seed)
    assignment: dict[int, str] = {}
    for label in sorted(members):
        group_sizes = members[label]
        if len(group_sizes) < 3:
            raise ValueError(
                f"Kelas {label!r} hanya punya {len(group_sizes)} kelompok; butuh minimal 3 "
                "agar train/validation/test masing-masing terisi."
            )
        ordered = sorted(group_sizes)
        rng.shuffle(ordered)
        total = sum(group_sizes.values())
        filled = Counter()
        for group in ordered:
            if filled["test"] < test_fraction * total:
                split = "test"
            elif filled["validation"] < validation_fraction * total:
                split = "validation"
            else:
                split = "train"
            assignment[group] = split
            filled[split] += group_sizes[group]
    return assignment


def drop_label_conflicts(
    records: list[CropRecord], groups: list[int], output_dir: Path
) -> tuple[list[CropRecord], list[int], list[CropRecord]]:
    labels_per_group: dict[int, set[str]] = defaultdict(set)
    for record, group in zip(records, groups):
        labels_per_group[group].add(record.label)
    conflicted = {group for group, labels in labels_per_group.items() if len(labels) > 1}
    kept, kept_groups, dropped = [], [], []
    for record, group in zip(records, groups):
        if group in conflicted:
            dropped.append(record)
            (output_dir / record.file).unlink(missing_ok=True)
        else:
            kept.append(record)
            kept_groups.append(group)
    return kept, kept_groups, dropped


# --------------------------------------------------------------------------- QA & laporan


def contact_sheet(
    tiles: Sequence[tuple[np.ndarray, str]], path: Path, columns: int = 8, tile: int = 150
) -> None:
    if not tiles:
        return
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.full((rows * (tile + 16), columns * tile, 3), 255, dtype=np.uint8)
    for position, (image, caption) in enumerate(tiles):
        row, column = divmod(position, columns)
        top, left = row * (tile + 16), column * tile
        sheet[top + 16 : top + 16 + tile, left : left + tile] = cv2.resize(
            image, (tile, tile), interpolation=cv2.INTER_AREA
        )
        cv2.putText(sheet, caption[:24], (left + 2, top + 11), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 0, 0), 1, cv2.LINE_AA)
    _write_jpeg(path, sheet)


def write_qa_sheets(
    records: Sequence[CropRecord],
    excluded: dict[str, list[tuple[Path, str]]],
    output_dir: Path,
    seed: int,
    samples: int = 48,
) -> None:
    qa = output_dir / "qa"
    qa.mkdir(exist_ok=True)
    rng = random.Random(seed)
    for label in ("healthy", "reject"):
        pool = [record for record in records if record.label == label]
        chosen = rng.sample(pool, min(samples, len(pool)))
        tiles = [
            (cv2.imdecode(np.fromfile(str(output_dir / record.file), dtype=np.uint8), cv2.IMREAD_COLOR),
             Path(record.file).stem)
            for record in chosen
        ]
        contact_sheet(tiles, qa / f"crops_{label}.jpg")
    for reason, items in excluded.items():
        tiles = []
        for path, label in rng.sample(items, min(24, len(items))):
            try:
                tiles.append((load_bgr(path, max_side=300), f"{label[:1].upper()}:{path.stem}"))
            except Exception:  # noqa: BLE001
                continue
        contact_sheet(tiles, qa / f"excluded_{reason}.jpg", columns=6, tile=180)


def build_report(
    parameters: dict[str, Any],
    source_counts: Counter,
    records: Sequence[CropRecord],
    excluded: dict[str, list[tuple[Path, str]]],
    conflicts: Sequence[CropRecord],
    groups: Sequence[int],
    assignment: dict[int, str],
) -> tuple[dict[str, Any], str]:
    labels = ("healthy", "reject")
    crop_counts = Counter(record.label for record in records)
    group_sizes = Counter(groups)
    largest = max(group_sizes.values()) if group_sizes else 0
    split_crops: Counter = Counter()
    split_groups_count: dict[tuple[str, str], set[int]] = defaultdict(set)
    for record, group in zip(records, groups):
        split_crops[(record.label, assignment[group])] += 1
        split_groups_count[(record.label, assignment[group])].add(group)
    solidity = [record.solidity for record in records]

    exclusion_rows = []
    for reason, description in REASONS.items():
        per_label = Counter(label for _, label in excluded.get(reason, []))
        if reason == "label_konflik":
            per_label = Counter(record.label for record in conflicts)
        exclusion_rows.append([reason, per_label["healthy"], per_label["reject"], description])
    summary = {
        "parameters": parameters,
        "sources": dict(source_counts),
        "crops": dict(crop_counts),
        "excluded": {r: len(v) for r, v in excluded.items()} | {"label_konflik": len(conflicts)},
        "groups": {"count": len(group_sizes), "largest": largest,
                   "largest_fraction": (largest / len(records)) if records else None},
        "split": {f"{label}/{split}": split_crops[(label, split)]
                  for label in labels for split in ("train", "validation", "test")},
        "solidity": {"min": float(min(solidity)) if solidity else None,
                     "median": float(np.median(solidity)) if solidity else None},
    }
    split_rows = [
        [label, split, split_crops[(label, split)], len(split_groups_count[(label, split)])]
        for label in labels for split in ("train", "validation", "test")
    ]
    markdown = "\n\n".join([
        "# Laporan persiapan dataset cacat v2",
        "## Parameter\n\n" + render_table(["parameter", "nilai"], [[k, str(v)] for k, v in parameters.items()]),
        "## Sumber dan hasil crop\n\n" + render_table(
            ["kelas", "gambar sumber", "crop dipakai"],
            [[label, source_counts[label], crop_counts[label]] for label in labels]),
        "## Gambar yang dikecualikan\n\n" + render_table(
            ["alasan", "healthy", "reject", "keterangan"], exclusion_rows)
        + "\n\nLihat contoh di folder `qa/` (excluded_<alasan>.jpg) untuk memastikan penyaringan wajar.",
        "## Split\n\n" + render_table(["kelas", "split", "crop", "kelompok"], split_rows),
        "## Kelompok\n\n" + render_table(
            ["ukuran", "nilai"],
            [["jumlah kelompok", len(group_sizes)], ["kelompok terbesar (crop)", largest],
             ["porsi kelompok terbesar", (largest / len(records)) if records else None],
             ["solidity minimum crop", summary["solidity"]["min"]],
             ["solidity median crop", summary["solidity"]["median"]]])
        + "\n\nBila kelompok terbesar lebih dari ~10% total crop, ambang kemiripan terlalu longgar "
          "(kelompok berantai): naikkan --dup-threshold.",
        "## Langkah berikutnya\n\n1. Buka `qa/crops_healthy.jpg` dan `qa/crops_reject.jpg`; hapus "
        "file crop yang salah label (manifest tetap valid).\n2. Latih dengan `--defect-data-dir` dan "
        "`--split-manifest` (lihat docs/DATASET_CACAT_V2.md).",
    ])
    return summary, markdown


# --------------------------------------------------------------------------- orkestrasi


def prepare(
    agml_dir: Path,
    output_dir: Path,
    size: int = 256,
    neutral_background: bool = True,
    multi_rotten: str = "exclude",
    group_block: int = 20,
    dup_threshold: float = 0.985,
    seed: int = 42,
    limit: int | None = None,
    overwrite: bool = False,
    qa_samples: int = 48,
) -> dict[str, Any]:
    if multi_rotten not in {"exclude", "include"}:
        raise ValueError("multi_rotten must be 'exclude' or 'include'.")
    output_dir = Path(output_dir)
    existing = [name for name in MANAGED_ENTRIES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{output_dir} sudah berisi hasil sebelumnya ({existing}). "
            "Pakai --output-dir lain atau --overwrite."
        )
    for name in existing:
        target = output_dir / name
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = list_sources(agml_dir, limit)
    logger.info("Memproses %d gambar sumber dari %s", len(sources), agml_dir)
    records, excluded, source_counts = process_sources(
        sources, output_dir, size, neutral_background, multi_rotten
    )
    if not records:
        raise ValueError("Tidak ada crop yang lolos penyaringan; periksa dataset dan ambang segmentasi.")
    groups = assign_groups(records, group_block, dup_threshold)
    records, groups, conflicts = drop_label_conflicts(records, groups, output_dir)
    if not records:
        raise ValueError("Semua crop berada di kelompok berlabel konflik.")
    assignment = split_groups(records, groups, seed)

    with (output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["file", "split", "group"])
        for record, group in sorted(zip(records, groups), key=lambda pair: pair[0].file):
            writer.writerow([record.file, assignment[group], f"g{group:05d}"])
    with (output_dir / "crops_index.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["file", "label", "source", "box_x", "box_y", "box_w", "box_h",
                         "area_fraction", "solidity", "regions_in_source"])
        for record in sorted(records, key=lambda item: item.file):
            writer.writerow([record.file, record.label, record.source, *record.box,
                             f"{record.area_fraction:.4f}", f"{record.solidity:.4f}",
                             record.regions_in_source])

    parameters = {
        "agml_dir": str(agml_dir), "size": size,
        "background": "neutral" if neutral_background else "keep",
        "multi_rotten": multi_rotten, "group_block": group_block,
        "dup_threshold": dup_threshold, "seed": seed, "limit": limit,
        "min_solidity": MIN_SOLIDITY,
    }
    summary, markdown = build_report(
        parameters, source_counts, records, excluded, conflicts, groups, assignment
    )
    (output_dir / "prepare_report.md").write_text(markdown + "\n", encoding="utf-8")
    (output_dir / "prepare_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_qa_sheets(records, excluded, output_dir, seed, qa_samples)
    logger.info("\n%s", markdown)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--agml-dir", required=True, type=Path, help="Folder berisi Fresh/ dan Rotten/.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--size", type=int, default=256, help="Sisi crop persegi (piksel).")
    parser.add_argument("--background", choices=["neutral", "keep"], default="neutral",
                        help="neutral: latar diganti warna netral; keep: latar asli dipertahankan.")
    parser.add_argument("--multi-rotten", choices=["exclude", "include"], default="exclude")
    parser.add_argument("--group-block", type=int, default=20,
                        help="Foto bernomor berdekatan (blok N) dianggap satu sesi; 0 = nonaktif.")
    parser.add_argument("--dup-threshold", type=float, default=0.985,
                        help="Korelasi minimum dua crop dianggap salinan nyaris identik; 0 = nonaktif.\n"
                             "Hanya menangkap salinan hasil simpan ulang/resize. Pada uji 18 tomat berbeda,\n"
                             "korelasi maksimum antar-tomat 0,93-0,94: jangan turunkan di bawah ~0,95.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Hanya N gambar pertama per kelas (uji cepat).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    prepare(args.agml_dir, args.output_dir, args.size, args.background == "neutral",
            args.multi_rotten, args.group_block, args.dup_threshold, args.seed, args.limit,
            args.overwrite)


if __name__ == "__main__":
    main()
