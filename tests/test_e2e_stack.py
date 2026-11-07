"""Uji nyata: controller / alat uji kamera -> HTTP -> dashboard (uvicorn sungguhan) -> database SQLite.

Hanya serial, kamera, dan model yang ditiru; dashboard, jalur HTTP, dan database adalah yang asli.
"""
import shutil
import socket
import threading
import time

import httpx
import pytest

from controller.config import load_config
from controller.dashboard_client import DashboardClient, check_dashboard
from controller.main import ROOT, Controller
from controller.serial_link import SerialLink
from dashboard.app import create_app
from tests.mock_serial import MockSerial
from tests.test_camera_predict import RESULT as CAMERA_RESULT
from tools.camera_predict import analyze, build_tomato_payload, format_summary
from tests.test_controller import FakeCamera, ML_RESULT

uvicorn = pytest.importorskip("uvicorn")
PHOTO_DIR = "data/debug_e2e_test"


@pytest.fixture
def stack(tmp_path, monkeypatch):
    monkeypatch.delenv("DASHBOARD_INGEST_KEY", raising=False)
    cfg = load_config()
    cfg["capture"]["settle_after_detect_s"] = 0
    cfg["dashboard"]["database"] = str(tmp_path / "e2e.db")
    cfg["paths"]["debug_dir"] = PHOTO_DIR
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cfg["dashboard"]["url"] = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(uvicorn.Config(create_app(cfg), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        try:
            if httpx.get(cfg["dashboard"]["url"] + "/api/health").status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.1)
    yield cfg
    server.should_exit = True
    thread.join(5)
    shutil.rmtree(ROOT / PHOTO_DIR, ignore_errors=True)


def test_full_sorting_cycle_ends_up_in_the_database(stack):
    base = stack["dashboard"]["url"]
    client = DashboardClient.from_config(stack).start()
    serial = MockSerial([l.encode() + b"\n" for l in ["READY", "DETECT,1", "ACK,1", "DONE,1,B", "READY"]])
    ctl = Controller(stack, SerialLink(serial_obj=serial), FakeCamera(True), lambda f: ("A", 0.9), client,
                     image_predict_fn=lambda jpeg: ML_RESULT, sleep=lambda s: None)
    for _ in range(5):
        ctl.step()
    client.frame(b"\xff\xd8\xff\xd9")
    client.close()

    assert serial.written == [b"GRADE,1,B\n"]
    (row,) = httpx.get(base + "/api/tomatoes").json()["rows"]
    assert (row["firmware_id"], row["grade"], row["serial_status"]) == (1, "B", "DONE")
    assert row["architecture"] == "resnext101" and row["ripeness_class"] == "setengah"
    assert httpx.get(f"{base}/api/tomatoes/{row['id']}/image").status_code == 200
    messages = [e["message"] for e in httpx.get(base + "/api/events").json()]
    assert any("terdeteksi" in m for m in messages) and any("selesai diantar" in m for m in messages)
    overview = httpx.get(base + "/api/overview").json()
    assert overview["stats"]["per_grade"] == {"A": 0, "B": 1, "C": 0}
    assert overview["state"]["serial_ok"] is True and overview["state"]["camera_ok"] is True
    assert httpx.get(base + "/api/frame").status_code == 200


def test_camera_test_result_reaches_the_dashboard_without_firmware_id(stack):
    base = stack["dashboard"]["url"]
    outcome = analyze(FakeCamera(True).read_fresh(), FakeCamera(False).read_fresh(), None, lambda jpeg: CAMERA_RESULT)
    client = DashboardClient.from_config(stack).start()

    client.status(serial_ok=False, camera_ok=True)
    client.tomato(build_tomato_payload(outcome, None))
    client.event("INFO", "controller", f"[Uji kamera] {format_summary(outcome)}")
    client.close()

    (row,) = httpx.get(base + "/api/tomatoes").json()["rows"]
    assert row["firmware_id"] is None and row["grade"] == "B" and row["architecture"] == "resnext101"
    state = httpx.get(base + "/api/overview").json()["state"]
    assert state["serial_ok"] is False and state["camera_ok"] is True
    assert any("[Uji kamera]" in e["message"] for e in httpx.get(base + "/api/events").json())


def test_check_dashboard_reports_live_and_dead_dashboard(stack):
    assert check_dashboard(stack["dashboard"]["url"])[0] is True
    ok, message = check_dashboard("http://127.0.0.1:9", timeout_s=0.5)
    assert not ok and "python -m dashboard" in message
