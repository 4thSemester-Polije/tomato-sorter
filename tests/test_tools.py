import csv

import cv2
import numpy as np

from tools.capture_dataset import PHOTO_FIELDS, append_photo_row, next_seq, photo_filename, tomato_id_str
from tools.extract_features_batch import FEATURE_FIELDS, process_photos


def test_naming():
    assert tomato_id_str(7) == "T007"
    assert photo_filename("T007", 3, "20261018") == "T007_03_20261018.jpg"


def test_next_seq_continues_from_existing(tmp_path):
    for name in ["T001_01_20261018.jpg", "T001_02_20261018.jpg", "T002_01_20261018.jpg"]:
        (tmp_path / name).write_bytes(b"x")
    assert next_seq(str(tmp_path), "T001") == 3
    assert next_seq(str(tmp_path), "T003") == 1


def test_append_photo_row_writes_header_once(tmp_path):
    p = str(tmp_path / "photos.csv")
    append_photo_row(p, {"id_tomat": "T001", "nama_file": "a.jpg", "tanggal": "2026-10-18"})
    append_photo_row(p, {"id_tomat": "T001", "nama_file": "b.jpg", "tanggal": "2026-10-18"})
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    assert len(rows) == 2 and list(rows[0].keys()) == PHOTO_FIELDS


def test_process_photos_end_to_end(tmp_path):
    img = np.full((480, 640, 3), (235, 235, 235), dtype=np.uint8)
    cv2.circle(img, (320, 240), 90, (30, 30, 200), -1)
    cv2.imwrite(str(tmp_path / "T001_01_20261018.jpg"), img)
    rows = [{"id_tomat": "T001", "nama_file": "T001_01_20261018.jpg"},
            {"id_tomat": "T002", "nama_file": "hilang.jpg"}]
    res = process_photos(rows, str(tmp_path), debug_dir=str(tmp_path / "debug"))
    assert res[0]["seg_ok"] == 1 and res[0]["pct_red"] > 90
    assert res[1]["seg_ok"] == 0
    assert set(res[0].keys()) == set(FEATURE_FIELDS)
    assert (tmp_path / "debug" / "T001_01_20261018.jpg").exists()
