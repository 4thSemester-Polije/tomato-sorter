import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import pytest

from controller.config import load_config
from controller.dashboard_client import DashboardClient
from controller.main import Controller, validate_config
from controller.ml_client import build_multipart, make_image_predictor
from controller.serial_link import SerialLink
from tests.mock_serial import MockSerial

BG = (235, 235, 235)
ML_RESULT = {"architecture": "resnext101", "grade": "B",
             "ripeness": {"class": "setengah", "level": 2, "confidence": 0.8, "probabilities": {}},
             "defect": {"class": "mulus", "level": 1, "confidence": 0.9, "probabilities": {}}}


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


class FakeClient:
    """Merekam semua yang dikirim controller ke dashboard (antarmuka sama dengan DashboardClient)."""

    def __init__(self):
        self.sent = []

    def tomato(self, p): self.sent.append(("tomato", p))
    def firmware(self, m): self.sent.append(("firmware", dict(m)))
    def event(self, level, source, message): self.sent.append(("event", level, source, message))
    def status(self, state=None, firmware_id=None, message=None, serial_ok=None, camera_ok=None):
        self.sent.append(("status", state, firmware_id, message))
    def frame(self, jpeg): self.sent.append(("frame", len(jpeg)))

    def of(self, kind): return [s for s in self.sent if s[0] == kind]


def make(tmp_path, incoming, predict=None, with_tomato=True, image_predict=None):
    cfg = load_config()
    cfg["capture"]["settle_after_detect_s"] = 0
    cfg["paths"]["debug_dir"] = str(tmp_path / "debug")
    serial = MockSerial([line.encode() + b"\n" for line in incoming])
    client = FakeClient()
    ctl = Controller(cfg, SerialLink(serial_obj=serial), FakeCamera(with_tomato), predict or (lambda f: ("A", 0.9)),
                     client, image_predict_fn=image_predict, sleep=lambda s: None)
    return ctl, serial, client


def run_all(ctl, n):
    for _ in range(n):
        ctl.step()


def test_full_cycle_sends_grade_and_forwards_everything(tmp_path):
    ctl, serial, client = make(tmp_path, ["READY", "DETECT,7", "ACK,7", "DONE,7,A", "READY"])
    run_all(ctl, 5)
    assert serial.written == [b"GRADE,7,A\n"]
    assert [m[1]["type"] for m in client.of("firmware")] == ["READY", "DETECT", "ACK", "DONE", "READY"]
    (_, t), = client.of("tomato")
    assert (t["firmware_id"], t["grade"]) == (7, "A") and t["latency_ms"] >= 0
    assert t["image_path"] and list((tmp_path / "debug").glob("T007_*.jpg"))
    assert ("status", "GRADING", 7, None) in client.of("status")


def test_tomato_posted_after_grade_sent_and_before_ack(tmp_path):
    ctl, _, client = make(tmp_path, ["DETECT,3", "ACK,3"])
    run_all(ctl, 2)
    kinds = [s[0] for s in client.sent if s[0] in ("firmware", "tomato")]
    assert kinds == ["firmware", "tomato", "firmware"]  # DETECT, hasil, ACK: dashboard sudah punya barisnya saat ACK tiba


def test_ml_api_result_forwarded_with_ripeness_and_defect(tmp_path):
    ctl, serial, client = make(tmp_path, ["DETECT,4"], image_predict=lambda jpeg: ML_RESULT)
    ctl.step()
    assert serial.written == [b"GRADE,4,B\n"]
    (_, t), = client.of("tomato")
    assert t["ripeness"]["class"] == "setengah" and t["defect"]["confidence"] == 0.9 and t["architecture"] == "resnext101"


def test_ml_api_failure_sends_nothing_and_logs_error(tmp_path):
    def down(jpeg):
        raise ConnectionRefusedError("mati")

    ctl, serial, client = make(tmp_path, ["DETECT,5"], image_predict=down)
    ctl.step()
    assert serial.written == [] and client.of("tomato") == []
    assert client.of("event")[0][1:3] == ("ERROR", "controller")


def test_no_valid_tomato_means_no_reply(tmp_path):
    ctl, serial, client = make(tmp_path, ["DETECT,1", "ERR,1,GRADE_TIMEOUT", "READY"], with_tomato=False)
    ctl.step()
    assert serial.written == []  # sesuai docs/ALUR_ALAT.md: PC tidak membalas
    assert client.of("event")[0][1] == "WARN"
    ctl.step()
    assert client.of("firmware")[-1][1] == {"type": "ERR", "id": 1, "code": "GRADE_TIMEOUT"}
    assert client.of("tomato") == []


def test_predict_not_implemented_sends_nothing(tmp_path, capsys):
    def boom(features):
        raise NotImplementedError

    ctl, serial, client = make(tmp_path, ["DETECT,2"], boom)
    ctl.step()
    assert serial.written == [] and client.of("tomato") == []
    assert "belum diimplementasikan" in capsys.readouterr().err


def test_idle_step_sends_heartbeat(tmp_path):
    ctl, _, client = make(tmp_path, [])
    ctl.step()
    assert client.of("status")[-1][1] == "READY" and client.of("firmware") == []


def test_validate_config_requires_settle_time():
    cfg = load_config()
    cfg["capture"]["settle_after_detect_s"] = None
    with pytest.raises(SystemExit):
        validate_config(cfg)


# ---------- klien HTTP nyata melawan server tiruan ----------
class Recorder(BaseHTTPRequestHandler):
    log = []
    status = 201

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        Recorder.log.append((self.path, self.headers.get("X-API-Key"), self.headers["Content-Type"], body))
        self.send_response(Recorder.status)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    Recorder.log, Recorder.status = [], 201
    srv = HTTPServer(("127.0.0.1", 0), Recorder)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_dashboard_client_posts_in_order_with_key(server):
    c = DashboardClient(server, "kunci").start()
    c.event("INFO", "controller", "satu")
    c.firmware({"type": "DONE", "id": 3, "grade": "A", "code": None})
    c.frame(b"\xff\xd8abc")
    c.close()
    paths = [p for p, *_ in Recorder.log]
    assert paths == ["/api/ingest/events", "/api/ingest/firmware", "/api/ingest/frame"]
    assert all(k == "kunci" for _, k, *_ in Recorder.log)
    assert json.loads(Recorder.log[1][3]) == {"type": "DONE", "id": 3, "grade": "A"}  # field None dibuang
    assert Recorder.log[2][2] == "image/jpeg"


def test_dashboard_client_keeps_only_latest_frame(server):
    c = DashboardClient(server)  # belum start: frame menumpuk
    for i in range(5):
        c.frame(bytes([0xFF, 0xD8, i]))
    c.start()
    c.close()
    frames = [b for p, _, _, b in Recorder.log if p == "/api/ingest/frame"]
    assert frames == [bytes([0xFF, 0xD8, 4])]


def test_dashboard_client_survives_dead_dashboard(capsys):
    c = DashboardClient("http://127.0.0.1:1", timeout_s=0.2).start()
    c.event("INFO", "controller", "tidak sampai")  # tidak boleh melempar atau memblokir
    c.close(flush_s=1)
    assert "tidak terjangkau" in capsys.readouterr().err


def test_ml_client_multipart_and_parse(monkeypatch):
    body, ctype = build_multipart(b"\xff\xd8xyz")
    boundary = ctype.split("boundary=")[1]
    assert body.startswith(f"--{boundary}".encode()) and b'name="file"' in body and body.endswith(f"--{boundary}--\r\n".encode())

    seen = {}

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return json.dumps(ML_RESULT).encode()

    def fake_urlopen(req, timeout):
        seen["url"], seen["timeout"] = req.full_url, timeout
        return Resp()

    monkeypatch.setattr("controller.ml_client.request.urlopen", fake_urlopen)
    assert make_image_predictor("http://ml:9000/", "resnext50", 3)(b"\xff\xd8")["grade"] == "B"
    assert seen == {"url": "http://ml:9000/predict?architecture=resnext50", "timeout": 3}


def test_ml_client_rejects_bad_response(monkeypatch):
    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return b'{"grade": "Z"}'

    monkeypatch.setattr("controller.ml_client.request.urlopen", lambda req, timeout: Resp())
    with pytest.raises(ValueError):
        make_image_predictor("http://ml")(b"x")
