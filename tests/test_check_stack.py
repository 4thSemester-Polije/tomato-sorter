import os

from controller.config import load_config
import controller.config as config_module
from tools.check_stack import FAIL, OK, WARN, format_checks, run_checks


def _run(cfg=None, **over):
    deps = dict(
        check_api=lambda url, arch: (True, "API model siap"),
        check_dashboard=lambda url: (True, "Dashboard terhubung"),
        serial_ports=lambda: [("COM3", "CH340")],
        camera_opens=lambda i: True,
        env={"GEMINI_API_KEY": "x"},
        exists=lambda p: True,
    )
    deps.update(over)
    return {c.name: c for c in run_checks(cfg or load_config(), **deps)}


def test_everything_ready():
    checks = _run()
    assert all(c.status == OK for c in checks.values())
    assert "Siap dijalankan" in format_checks(list(checks.values()))


def test_failures_block_but_warnings_do_not():
    checks = _run(check_api=lambda u, a: (False, "API mati"), serial_ports=lambda: [("COM9", "x")],
                  camera_opens=lambda i: False, check_dashboard=lambda u: (False, "dashboard mati"),
                  exists=lambda p: False, env={})
    assert checks["API model"].status == FAIL
    assert checks["Port serial"].status == FAIL and "COM9" in checks["Port serial"].detail
    assert checks["Kamera"].status == FAIL
    assert checks["Dashboard"].status == WARN
    assert checks["Foto latar kosong"].status == WARN
    assert checks["GEMINI_API_KEY (chatbot)"].status == WARN
    text = format_checks(list(checks.values()))
    assert "3 pemeriksaan GAGAL" in text and "peringatan" in text


def test_null_api_url_and_bad_config_are_failures():
    cfg = load_config()
    cfg["ml"]["api_url"] = None
    cfg["capture"]["settle_after_detect_s"] = "1.0."
    checks = _run(cfg)
    assert checks["API model"].status == FAIL and checks["Config"].status == FAIL


def test_load_env_reads_dotenv_but_does_not_override_terminal(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("a: 1\n")
    (tmp_path / ".env").write_text("DASHBOARD_INGEST_KEY=dari-dotenv\nGEMINI_API_KEY=dari-dotenv\n")
    monkeypatch.setattr(config_module, "DEFAULT_PATH", tmp_path / "config.yaml")
    monkeypatch.delenv("DASHBOARD_INGEST_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "dari-terminal")

    config_module.load_env()

    assert os.environ["DASHBOARD_INGEST_KEY"] == "dari-dotenv"
    assert os.environ["GEMINI_API_KEY"] == "dari-terminal"
    monkeypatch.delenv("DASHBOARD_INGEST_KEY", raising=False)
