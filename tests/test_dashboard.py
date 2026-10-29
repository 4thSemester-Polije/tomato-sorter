import time

import pytest

from controller.config import load_config
from dashboard import db
from vision.features import FEATURE_NAMES

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from dashboard.app import create_app  # noqa: E402

FEATS = {"pct_red": 70.0, "mean_h": 5.0, "mean_s": 150.0, "mean_v": 140.0, "pct_dark": 1.5, "diameter_px": 170.0, "diameter_mm": None}


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "t.db"
    db.init_db(path)
    return path


@pytest.fixture
def client(tmp_path):
    cfg = load_config()
    cfg["dashboard"]["database"] = str(tmp_path / "t.db")
    cfg["dashboard"]["live_frame"] = str(tmp_path / "live.jpg")
    return TestClient(create_app(cfg, root=tmp_path)), tmp_path


def test_feature_columns_match_vision_contract():
    assert db.FEATURE_COLUMNS == FEATURE_NAMES  


def test_insert_requires_exactly_grade_or_error(database):
    with pytest.raises(ValueError):
        db.insert_result(database, 1)
    with pytest.raises(ValueError):
        db.insert_result(database, 1, grade="A", err_code="BUSY")
    with pytest.raises(ValueError):
        db.insert_result(database, 1, grade="D")
    with pytest.raises(ValueError):
        db.insert_result(database, 1, err_code="NGAWUR")
    with pytest.raises(ValueError):
        db.insert_result(database, 1, grade="A", features={"warna_ajaib": 1})


def test_set_status_keeps_connection_flags_when_omitted(database):
    db.set_status(database, "READY", serial_ok=True, kamera_ok=True, now=100)
    db.set_status(database, "DETECT", id_tomat=3, now=101)
    s = db.get_status(database)
    assert (s["state"], s["id_tomat"], s["serial_ok"], s["kamera_ok"], s["diperbarui"]) == ("DETECT", 3, 1, 1, 101)
    with pytest.raises(ValueError):
        db.set_status(database, "MELAYANG")


def test_stats_counts_and_throughput_buckets(database):
    now = 1_000_000 * 60  # tepat di awal sebuah menit
    db.insert_result(database, 1, features=FEATS, grade="A", latensi_s=1.0, waktu=now)
    db.insert_result(database, 2, features=FEATS, grade="A", latensi_s=2.0, waktu=now - 30)
    db.insert_result(database, 3, features=FEATS, grade="C", waktu=now - 60 * 3)
    db.insert_result(database, 4, err_code="GRADE_TIMEOUT", waktu=now - 10)
    db.insert_result(database, 5, features=FEATS, grade="B", waktu=now - 60 * 100)  # di luar jendela
    st = db.get_stats(database, window_min=5, now=now)
    assert st["total"] == 4
    assert st["per_grade"] == {"A": 2, "B": 1, "C": 1}
    assert st["errors"] == {"GRADE_TIMEOUT": 1} and st["error_total"] == 1
    assert st["latensi_rata2_s"] == pytest.approx(1.5)
    assert len(st["per_menit"]) == 5
    assert st["per_menit"][-1] == 1  # menit sekarang
    assert st["per_menit"][-2] == 1  # 30 detik lalu jatuh ke menit sebelumnya
    assert st["per_menit"][-4] == 1  # 3 menit lalu
    assert sum(st["per_menit"]) == 3


def test_history_filter_and_paging(database):
    for i in range(1, 8):
        db.insert_result(database, i, features=FEATS, grade="A" if i % 2 else "B")
    db.insert_result(database, 8, err_code="BUSY")
    rows, total = db.list_history(database, limit=3, offset=0)
    assert total == 8 and [r["id_tomat"] for r in rows] == [8, 7, 6]
    rows, total = db.list_history(database, hasil="A")
    assert total == 4 and all(r["grade"] == "A" for r in rows)
    rows, total = db.list_history(database, hasil="err")
    assert total == 1 and rows[0]["err_code"] == "BUSY"
    with pytest.raises(ValueError):
        db.list_history(database, hasil="Z")


def test_export_csv_header_and_rows(database):
    db.insert_result(database, 1, features=FEATS, grade="A", confidence=0.9)
    lines = db.export_csv(database).strip().split("\n")
    assert lines[0].split(",") == list(db.CSV_COLUMNS)
    assert len(lines) == 2 and ",A,0.9," in lines[1]


def test_overview_marks_status_stale_without_heartbeat(client):
    c, tmp = client
    r = c.get("/api/overview").json()
    assert r["status"]["usang"] is True  # belum pernah ada kabar
    assert r["terakhir"] is None and r["frame"] == {"ada": False, "umur_s": None}
    db.set_status(tmp / "t.db", "READY", serial_ok=True, kamera_ok=True)
    assert c.get("/api/overview").json()["status"]["usang"] is False


def test_history_endpoint_validates_filter(client):
    c, tmp = client
    db.insert_result(tmp / "t.db", 1, features=FEATS, grade="B")
    assert c.get("/api/history?hasil=B").json()["total"] == 1
    assert c.get("/api/history?hasil=Z").status_code == 422
    assert c.get("/api/history?limit=0").status_code == 422


def test_frame_endpoint(client):
    c, tmp = client
    assert c.get("/api/frame").status_code == 404
    (tmp / "live.jpg").write_bytes(b"\xff\xd8fake\xff\xd9")
    r = c.get("/api/frame")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert c.get("/api/overview").json()["frame"]["ada"] is True


def test_foto_only_served_from_data_dir(client):
    c, tmp = client
    (tmp / "data").mkdir()
    (tmp / "data" / "a.jpg").write_bytes(b"jpeg")
    (tmp / "rahasia.txt").write_text("jangan bocor")
    ok = db.insert_result(tmp / "t.db", 1, features=FEATS, grade="A", foto="data/a.jpg")
    bad = db.insert_result(tmp / "t.db", 2, features=FEATS, grade="A", foto="rahasia.txt")
    trav = db.insert_result(tmp / "t.db", 3, features=FEATS, grade="A", foto="data/../rahasia.txt")
    none = db.insert_result(tmp / "t.db", 4, features=FEATS, grade="A")
    assert c.get(f"/api/foto/{ok}").status_code == 200
    assert c.get(f"/api/foto/{bad}").status_code == 404
    assert c.get(f"/api/foto/{trav}").status_code == 404
    assert c.get(f"/api/foto/{none}").status_code == 404


def test_index_and_static_served(client):
    c, _ = client
    assert "Stasiun pemilah tomat" in c.get("/").text
    assert c.get("/static/app.js").status_code == 200
    assert c.get("/api/export.csv").headers["content-type"].startswith("text/csv")


def test_index_and_static_404(client):
    c, _ = client
    assert c.get("/static/does-not-exist.js").status_code == 404
    assert c.get("/does-not-exist").status_code == 404
