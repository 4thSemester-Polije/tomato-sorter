import io
import json

import pytest

from controller import ml_client
from controller.config import load_config
from controller.main import validate_config
from tools.hardware_check import format_report


# ---- config.local.yaml ----
def test_local_config_overrides_only_given_keys(tmp_path):
    (tmp_path / "config.yaml").write_text("serial:\n  port: COM3\n  baudrate: 115200\ncamera:\n  index: 0\n")
    (tmp_path / "config.local.yaml").write_text("serial:\n  port: COM7\ncamera:\n  index: 1\n")

    cfg = load_config(tmp_path / "config.yaml")

    assert cfg["serial"] == {"port": "COM7", "baudrate": 115200}
    assert cfg["camera"]["index"] == 1


def test_config_without_local_file_is_unchanged(tmp_path):
    (tmp_path / "config.yaml").write_text("serial:\n  port: COM3\n")

    assert load_config(tmp_path / "config.yaml") == {"serial": {"port": "COM3"}}


def test_empty_local_file_is_ignored_and_non_mapping_is_rejected(tmp_path):
    (tmp_path / "config.yaml").write_text("a: 1\n")
    (tmp_path / "config.local.yaml").write_text("")
    assert load_config(tmp_path / "config.yaml") == {"a": 1}

    (tmp_path / "config.local.yaml").write_text("- bukan\n- pemetaan\n")
    with pytest.raises(ValueError):
        load_config(tmp_path / "config.yaml")


def test_repo_config_loads_with_numeric_settle():
    cfg = load_config()
    validate_config(cfg)  # tidak boleh melempar: `1.0.` (string) dulu lolos null-check tapi merusak time.sleep


# ---- validate_config ----
@pytest.mark.parametrize("bad", ["1.0.", "satu", True, -1])
def test_validate_config_rejects_non_numeric_or_negative_settle(bad):
    with pytest.raises(SystemExit):
        validate_config({"capture": {"settle_after_detect_s": bad}})


@pytest.mark.parametrize("good", [0, 1, 1.5])
def test_validate_config_accepts_numbers(good):
    validate_config({"capture": {"settle_after_detect_s": good}})


# ---- check_api ----
class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _serve(monkeypatch, body):
    monkeypatch.setattr(ml_client.request, "urlopen", lambda url, timeout=0: _Resp(json.dumps(body).encode()))


def test_check_api_ready_for_requested_architecture(monkeypatch):
    _serve(monkeypatch, {"architecture": "resnext101", "status": "ready",
                         "architectures": {"resnext101": {"status": "ready", "missing_files": []},
                                           "resnext50": {"status": "model_missing", "missing_files": ["x"]}}})

    ok, message = ml_client.check_api("http://127.0.0.1:9000", "resnext101")

    assert ok and "resnext101" in message


def test_check_api_reports_missing_model_files(monkeypatch):
    _serve(monkeypatch, {"architecture": "resnext101", "status": "ready",
                         "architectures": {"resnext50": {"status": "model_missing",
                                                         "missing_files": ["ripeness_resnext50.keras"]}}})

    ok, message = ml_client.check_api("http://127.0.0.1:9000", "resnext50")

    assert not ok and "ripeness_resnext50.keras" in message


def test_check_api_uses_server_default_when_architecture_is_null(monkeypatch):
    _serve(monkeypatch, {"architecture": "resnext101", "status": "ready",
                         "architectures": {"resnext101": {"status": "ready", "missing_files": []}}})

    assert ml_client.check_api("http://127.0.0.1:9000", None)[0] is True


def test_check_api_unreachable(monkeypatch):
    def boom(url, timeout=0):
        raise OSError("refused")

    monkeypatch.setattr(ml_client.request, "urlopen", boom)

    ok, message = ml_client.check_api("http://127.0.0.1:9000")

    assert not ok and "uvicorn ml.api:app" in message


# ---- hardware_check ----
def test_format_report_suggests_first_detected_devices():
    text = format_report([("COM5", "USB-SERIAL CH340")], [(1, (640, 480))])

    assert "COM5" in text and "indeks 1" in text and "config.local.yaml" in text
    assert "port: COM5" in text and "index: 1" in text


def test_format_report_handles_no_devices():
    text = format_report([], [])

    assert "tidak ada" in text and "port: COM3" in text and "index: 0" in text
