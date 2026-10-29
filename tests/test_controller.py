import cv2
import numpy as np
import pytest

from controller.config import load_config
from controller.main import Controller, validate_config
from controller.serial_link import SerialLink
from dashboard import db
from tests.mock_serial import MockSerial

BG = (235, 235, 235)


class FakeCamera:
    """Kamera tiruan: stasiun kosong atau tomat merah di tengah."""

    def __init__(self, with_tomato=True):
        self.with_tomato = with_tomato

    def _frame(self):
        img = np.full((480, 640, 3), BG, dtype=np.uint8)
        if self.with_tomato:
            cv2.circle(img, (320, 240), 90, (30, 30, 200), -1)
        return img

    read = read_fresh = _frame


def make(tmp_path, incoming, predict, with_tomato=True):
    cfg = load_config()
    cfg["capture"]["settle_after_detect_s"] = 0
    cfg["paths"]["debug_dir"] = str(tmp_path / "debug")
    database = tmp_path / "t.db"
    db.init_db(database)
    serial = MockSerial([line.encode() + b"\n" for line in incoming])
    ctl = Controller(cfg, SerialLink(serial_obj=serial), FakeCamera(with_tomato), predict, database, sleep=lambda s: None)
    return ctl, serial, database


def run_all(ctl, n):
    for _ in range(n):
        ctl.step()


def test_full_cycle_records_result_and_sends_grade(tmp_path):
    ctl, serial, database = make(tmp_path, ["READY", "DETECT,7", "ACK,7", "DONE,7,A", "READY"], lambda f: ("A", 0.9))
    run_all(ctl, 5)
    assert serial.written == [b"GRADE,7,A\n"]
    rows, total = db.list_history(database)
    assert total == 1
    r = rows[0]
    assert (r["id_tomat"], r["grade"], r["confidence"]) == (7, "A", 0.9)
    assert r["pct_red"] > 50 and r["latensi_s"] >= 0
    assert r["foto"] and (tmp_path / "debug").exists() and list((tmp_path / "debug").glob("T007_*.jpg"))
    assert db.get_status(database)["state"] == "READY"


def test_status_follows_protocol_messages(tmp_path):
    ctl, _, database = make(tmp_path, ["DETECT,3", "ACK,3", "DONE,3,B"], lambda f: ("B", 0.8))
    ctl.step()
    assert db.get_status(database)["state"] == "GRADING" and db.get_status(database)["id_tomat"] == 3
    ctl.step()
    assert db.get_status(database)["state"] == "ACK"
    ctl.step()
    assert db.get_status(database)["state"] == "DONE"


def test_no_valid_tomato_means_no_reply(tmp_path):
    ctl, serial, database = make(tmp_path, ["DETECT,1", "ERR,1,GRADE_TIMEOUT", "READY"], lambda f: ("A", 0.9), with_tomato=False)
    ctl.step()
    assert serial.written == []  # sesuai docs/ALUR_ALAT.md: PC tidak membalas
    ctl.step()
    rows, _ = db.list_history(database)
    assert rows[0]["err_code"] == "GRADE_TIMEOUT" and rows[0]["grade"] is None
    st = db.get_status(database)
    assert st["state"] == "ERR" and st["pesan"] == "GRADE_TIMEOUT"


def test_predict_not_implemented_sends_nothing(tmp_path, capsys):
    def boom(features):
        raise NotImplementedError

    ctl, serial, database = make(tmp_path, ["DETECT,2"], boom)
    ctl.step()
    assert serial.written == []
    assert "belum diimplementasikan" in capsys.readouterr().err


def test_done_without_pc_grade_is_ignored(tmp_path):
    ctl, _, database = make(tmp_path, ["DONE,9,C"], lambda f: ("C", 0.5))
    ctl.step()
    assert db.list_history(database)[1] == 0  # uji manual lewat serial monitor tidak dicatat


def test_idle_step_sends_heartbeat(tmp_path):
    ctl, _, database = make(tmp_path, [], lambda f: ("A", 1.0))
    ctl.step()  # tidak ada pesan
    st = db.get_status(database)
    assert st["serial_ok"] == 1 and st["diperbarui"] > 0


def test_validate_config_requires_settle_time():
    cfg = load_config()
    cfg["capture"]["settle_after_detect_s"] = None
    with pytest.raises(SystemExit):
        validate_config(cfg)