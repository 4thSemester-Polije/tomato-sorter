import json
from types import SimpleNamespace

import pytest

from controller.config import load_config

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from dashboard.app import create_app  # noqa: E402
from dashboard.chat import ChatUnavailable, GeminiClient  # noqa: E402
from dashboard.chat_tools import TOOLS, run_tool  # noqa: E402

KEY = {"X-API-Key": "rahasia-tes"}
JPEG = b"\xff\xd8\xff\xe0" + b"0" * 50
PRED = {  # bentuk mengikuti respons POST /predict di ml/api.py (ada field ekstra yang harus diabaikan)
    "model": "ResNeXt-101", "architecture": "resnext101", "input_size": [224, 224],
    "ripeness": {"class": "matang", "level": 1, "confidence": 0.93, "probabilities": {"matang": 0.93}},
    "defect": {"class": "mulus", "level": 1, "confidence": 0.88, "probabilities": {"mulus": 0.88}},
    "grade": "A",
}


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("DASHBOARD_INGEST_KEY", KEY["X-API-Key"])


def make_app(tmp_path, chat_client_factory=None):
    cfg = load_config()
    cfg["dashboard"]["database"] = str(tmp_path / "t.db")
    kw = {"chat_client_factory": chat_client_factory} if chat_client_factory else {}
    return create_app(cfg, root=tmp_path, **kw)


@pytest.fixture
def client(tmp_path):
    with TestClient(make_app(tmp_path)) as c:
        yield c


def post_tomato(client, **over):
    body = {"firmware_id": 7, "grade": "A", "latency_ms": 1200, "architecture": "resnext101", **over}
    r = client.post("/api/ingest/tomatoes", json=body, headers=KEY)
    assert r.status_code == 201, r.text
    return r.json()


# ---------- keamanan ingest ----------
def test_ingest_needs_key(client):
    assert client.post("/api/ingest/events", json={"level": "INFO", "source": "api", "message": "x"}).status_code == 401
    r = client.post("/api/ingest/events", json={"level": "INFO", "source": "api", "message": "x"}, headers={"X-API-Key": "salah"})
    assert r.status_code == 401


def test_without_key_configured_only_loopback_allowed(tmp_path, monkeypatch):
    monkeypatch.delenv("DASHBOARD_INGEST_KEY")
    with TestClient(make_app(tmp_path)) as c:  # klien tes bukan loopback
        assert c.post("/api/ingest/events", json={"level": "INFO", "source": "api", "message": "x"}).status_code == 403
    with TestClient(make_app(tmp_path), client=("127.0.0.1", 5000)) as c:
        assert c.post("/api/ingest/events", json={"level": "INFO", "source": "api", "message": "x"}).status_code == 201


# ---------- hasil model ----------
def test_ingest_accepts_ml_api_response_shape(client):
    t = post_tomato(client, ripeness=PRED["ripeness"], defect=PRED["defect"], image_path="data/debug/a.jpg")
    assert (t["ripeness_class"], t["ripeness_confidence"], t["defect_class"], t["defect_confidence"]) == ("matang", 0.93, "mulus", 0.88)
    assert t["serial_status"] is None and t["waktu"].endswith("Z")


def test_ingest_validates(client):
    assert client.post("/api/ingest/tomatoes", json={"grade": "D"}, headers=KEY).status_code == 422
    bad = {"grade": "A", "ripeness": {"class": "x", "confidence": 1.5}}
    assert client.post("/api/ingest/tomatoes", json=bad, headers=KEY).status_code == 422


# ---------- firmware ----------
def fw(client, **msg):
    r = client.post("/api/ingest/firmware", json=msg, headers=KEY)
    assert r.status_code == 200, r.text
    return r.json()


def test_firmware_cycle_updates_serial_status_and_state(client):
    t = post_tomato(client, firmware_id=12)
    fw(client, type="ACK", id=12)
    assert client.get(f"/api/tomatoes/{t['id']}").json()["serial_status"] == "ACK"
    assert client.get("/api/overview").json()["state"]["state"] == "ACK"
    fw(client, type="DONE", id=12, grade="A")
    assert client.get(f"/api/tomatoes/{t['id']}").json()["serial_status"] == "DONE"
    fw(client, type="ACK", id=12)  # ACK terlambat tidak menurunkan status final
    assert client.get(f"/api/tomatoes/{t['id']}").json()["serial_status"] == "DONE"


def test_firmware_err_logs_error_and_marks_tomato(client):
    t = post_tomato(client, firmware_id=3)
    fw(client, type="ERR", id=3, code="JAM")
    assert client.get(f"/api/tomatoes/{t['id']}").json()["serial_status"] == "ERR"
    errs = client.get("/api/events", params={"level": "ERROR"}).json()
    assert errs[0]["source"] == "firmware" and "JAM" in errs[0]["message"]
    st = client.get("/api/overview").json()
    assert st["state"]["state"] == "ERR" and st["state"]["message"] == "JAM" and st["stats"]["error_count"] == 1


def test_done_grade_mismatch_warns(client):
    post_tomato(client, firmware_id=5, grade="A")
    fw(client, type="DONE", id=5, grade="C")
    warns = client.get("/api/events", params={"level": "WARN"}).json()
    assert len(warns) == 1 and "grade A" in warns[0]["message"]


def test_firmware_message_validation(client):
    for bad in ({"type": "DONE", "id": 1}, {"type": "ERR", "id": 1}, {"type": "ACK"}, {"type": "NOPE"}):
        assert client.post("/api/ingest/firmware", json=bad, headers=KEY).status_code == 422
    fw(client, type="READY")


def test_same_firmware_id_after_reboot_matches_latest_row(client):
    a = post_tomato(client, firmware_id=1)
    fw(client, type="DONE", id=1, grade="A")
    b = post_tomato(client, firmware_id=1)  # penghitung firmware mulai ulang
    fw(client, type="ACK", id=1)
    assert client.get(f"/api/tomatoes/{a['id']}").json()["serial_status"] == "DONE"
    assert client.get(f"/api/tomatoes/{b['id']}").json()["serial_status"] == "ACK"


# ---------- kueri ----------
def test_stats_history_filter_and_csv(client):
    for g, ms in (("A", 1000), ("A", 2000), ("B", 3000), ("C", None)):
        post_tomato(client, grade=g, latency_ms=ms)
    s = client.get("/api/overview").json()["stats"]
    assert s["per_grade"] == {"A": 2, "B": 1, "C": 1} and s["total"] == 4
    assert s["avg_latency_ms"] == 2000 and s["per_minute"][-1] == 4 and len(s["per_minute"]) == 15
    page = client.get("/api/tomatoes", params={"grade": "A", "limit": 1}).json()
    assert page["total"] == 2 and len(page["rows"]) == 1
    assert client.get("/api/tomatoes", params={"grade": "Z"}).status_code == 422
    csv_text = client.get("/api/export.csv").text.splitlines()
    assert csv_text[0].startswith("id,waktu,firmware_id,grade") and len(csv_text) == 5


def test_csv_blocks_formula_injection(client):
    post_tomato(client, ripeness={"class": "=HYPERLINK(1)", "confidence": 0.5})
    assert "'=HYPERLINK(1)" in client.get("/api/export.csv").text


def test_image_served_only_from_data_dir(client, tmp_path):
    (tmp_path / "data" / "debug").mkdir(parents=True)
    (tmp_path / "data" / "debug" / "a.jpg").write_bytes(JPEG)
    (tmp_path / "rahasia.jpg").write_bytes(JPEG)
    ok = post_tomato(client, image_path="data/debug/a.jpg")
    bad = post_tomato(client, image_path="rahasia.jpg")
    trav = post_tomato(client, image_path="data/../rahasia.jpg")
    assert client.get(f"/api/tomatoes/{ok['id']}/image").status_code == 200
    assert client.get(f"/api/tomatoes/{bad['id']}/image").status_code == 404
    assert client.get(f"/api/tomatoes/{trav['id']}/image").status_code == 404


# ---------- kamera ----------
def test_frame_ingest_and_snapshot(client):
    assert client.get("/api/frame").status_code == 404
    assert client.post("/api/ingest/frame", content=b"bukan jpeg", headers=KEY).status_code == 415
    assert client.post("/api/ingest/frame", content=JPEG, headers=KEY).status_code == 204
    r = client.get("/api/frame")
    assert r.status_code == 200 and r.content == JPEG and client.get("/api/overview").json()["has_frame"] is True


def test_frame_size_limit(tmp_path):
    cfg = load_config()
    cfg["dashboard"].update(database=str(tmp_path / "t.db"), max_frame_bytes=100)
    with TestClient(create_app(cfg, root=tmp_path)) as c:
        assert c.post("/api/ingest/frame", content=JPEG + b"0" * 200, headers=KEY).status_code == 413


def test_hub_frames_yields_newest(tmp_path):
    import asyncio

    from dashboard.hub import Hub

    async def go():
        hub = Hub()
        await hub.publish_frame(b"\xff\xd8one")
        gen = hub.frames(1000)
        first = await anext(gen)
        await hub.publish_frame(b"\xff\xd8two")
        second = await anext(gen)
        return first, second

    assert asyncio.run(go()) == (b"\xff\xd8one", b"\xff\xd8two")


def test_hub_stream_waits_when_no_frame_yet():
    import asyncio

    from dashboard.hub import Hub

    async def go():
        hub = Hub()
        gen = hub.frames(1000)
        waiter = asyncio.ensure_future(anext(gen))  # penonton datang sebelum controller mengirim frame pertama
        await asyncio.sleep(0.05)
        assert not waiter.done()
        await hub.publish_frame(b"\xff\xd8pertama")
        return await asyncio.wait_for(waiter, 2)

    assert asyncio.run(go()) == b"\xff\xd8pertama"


# ---------- WebSocket ----------
def test_websocket_snapshot_then_live_push(client):
    post_tomato(client, grade="B")
    with client.websocket_connect("/ws") as ws:
        snap = ws.receive_json()
        assert snap["type"] == "snapshot" and snap["tomatoes"][0]["grade"] == "B" and snap["state"]["stale"] is True
        post_tomato(client, grade="A")
        msg = ws.receive_json()
        assert msg["type"] == "tomato" and msg["tomato"]["grade"] == "A" and msg["stats"]["total"] == 2
        fw(client, type="ERR", id=9, code="BUSY")
        types = {ws.receive_json()["type"] for _ in range(4)}
        assert {"state", "event", "stats"} <= types


def test_status_heartbeat_clears_stale(client):
    assert client.get("/api/overview").json()["state"]["stale"] is True
    client.post("/api/ingest/status", json={"state": "GRADING", "firmware_id": 4, "serial_ok": True, "camera_ok": True}, headers=KEY)
    st = client.get("/api/overview").json()["state"]
    assert st["stale"] is False and st["state"] == "GRADING" and st["camera_ok"] is True
    client.post("/api/ingest/status", json={"camera_ok": False}, headers=KEY)  # heartbeat polos tidak mengubah state
    st = client.get("/api/overview").json()["state"]
    assert st["state"] == "GRADING" and st["camera_ok"] is False and st["serial_ok"] is True


# ---------- chatbot ----------
def test_gemini_client_sends_server_key_and_function_tools(monkeypatch):
    seen = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"candidates": []}

    def fake_post(url, **kwargs):
        seen.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr("dashboard.chat.httpx.post", fake_post)
    GeminiClient("test-api-key").generate_content(
        model="gemini-test",
        contents=[{"role": "user", "parts": [{"text": "grade tomat?"}]}],
        system_instruction="sistem",
        tools=[{"name": "grade_summary", "parametersJsonSchema": {"type": "object"}}],
        max_tokens=123,
    )
    assert seen["url"].endswith("/models/gemini-test:generateContent")
    assert seen["headers"] == {
        "x-goog-api-key": "test-api-key",
        "Content-Type": "application/json",
    }
    assert seen["json"]["systemInstruction"]["parts"][0]["text"] == "sistem"
    declaration = seen["json"]["tools"][0]["functionDeclarations"][0]
    assert declaration["name"] == "grade_summary"
    assert declaration["parametersJsonSchema"] == {"type": "object"}
    assert seen["json"]["generationConfig"] == {"maxOutputTokens": 123}


class FakeGemini:
    """Klien Gemini tiruan: putaran 1 minta fungsi, putaran 2 menjawab dari hasil fungsi."""

    def __init__(self, tool_name="grade_summary", tool_input=None):
        self.calls = []
        self.tool_name, self.tool_input = tool_name, tool_input or {}

    def generate_content(self, **kw):
        self.calls.append(kw)
        last_parts = kw["contents"][-1]["parts"]
        if "functionResponse" in last_parts[0]:  # hasil fungsi sudah ada
            return {"candidates": [{"content": {"parts": [{"text": "Hasil: " + json.dumps(last_parts[0]["functionResponse"]["response"])}]}}]}
        return {"candidates": [{"content": {"parts": [{"functionCall": {"name": self.tool_name, "args": self.tool_input}}]}}]}


def test_chat_uses_tool_and_persists_history(tmp_path):
    fake = FakeGemini()
    with TestClient(make_app(tmp_path, lambda: fake)) as c:
        post_tomato(c, grade="A")
        post_tomato(c, grade="B")
        r = c.post("/api/chat", json={"message": "berapa tomat per grade?"})
        assert r.status_code == 200
        body = r.json()
        assert '"A": 1' in body["reply"] and '"total": 2' in body["reply"]
        hist = c.get(f"/api/chat/{body['session_id']}").json()
        assert [m["role"] for m in hist] == ["user", "assistant"]
        c.post("/api/chat", json={"session_id": body["session_id"], "message": "lagi"})
        assert len(c.get(f"/api/chat/{body['session_id']}").json()) == 4
    first = fake.calls[0]
    assert {t["name"] for t in first["tools"]} == {"grade_summary", "latest_tomatoes", "recent_errors", "average_latency"}
    assert "hanya bisa MEMBACA" in first["system_instruction"] and first["model"] == "gemini-3.6-flash"
    assert all(
        "minimum" not in prop and "maximum" not in prop
        for tool in first["tools"]
        for prop in tool["parametersJsonSchema"]["properties"].values()
    )
    assert all(tool["parametersJsonSchema"]["type"] == "object" for tool in first["tools"])
    assert all(
        prop["type"] == "integer"
        for tool in first["tools"]
        for prop in tool["parametersJsonSchema"]["properties"].values()
    )


def test_chat_without_api_key_is_503_and_never_leaks(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with TestClient(make_app(tmp_path)) as c:
        r = c.post("/api/chat", json={"message": "berapa grade tomat?"})
    assert r.status_code == 503 and "GEMINI_API_KEY" in r.json()["detail"]


def test_chat_upstream_failure_hides_details(tmp_path):
    class Boom:
        generate_content = lambda self, **kw: (_ for _ in ()).throw(RuntimeError("AIza-Rahasia"))

    with TestClient(make_app(tmp_path, lambda: Boom())) as c:
        r = c.post("/api/chat", json={"message": "berapa grade tomat?"})
    assert r.status_code == 503 and "Rahasia" not in r.text
    assert "kesalahan saat memproses" in r.json()["detail"]


@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (400, "Permintaan Gemini ditolak"),
        (401, "API key Gemini tidak valid"),
        (403, "Akses Gemini ditolak"),
        (404, "Model Gemini"),
        (429, "Kuota Gemini habis"),
        (500, "Layanan Gemini sedang bermasalah"),
    ],
)
def test_chat_explains_gemini_http_errors(tmp_path, status_code, message):
    import httpx

    def fail(**_kwargs):
        request = httpx.Request("POST", "https://example.test/generate")
        response = httpx.Response(status_code, request=request)
        raise httpx.HTTPStatusError("upstream detail must not leak", request=request, response=response)

    with TestClient(make_app(tmp_path, lambda: SimpleNamespace(generate_content=fail))) as c:
        r = c.post("/api/chat", json={"message": "berapa grade tomat?"})
    assert r.status_code == 503
    assert message in r.json()["detail"]
    assert "upstream detail" not in r.text


def test_chat_shows_sanitized_gemini_bad_request_detail(tmp_path):
    import httpx

    secret = "AIza" + "A" * 30

    def fail(**_kwargs):
        request = httpx.Request("POST", "https://example.test/generate")
        response = httpx.Response(
            400,
            request=request,
            json={"error": {"message": f"Invalid request using {secret}"}},
        )
        raise httpx.HTTPStatusError("upstream detail", request=request, response=response)

    with TestClient(make_app(tmp_path, lambda: SimpleNamespace(generate_content=fail))) as c:
        r = c.post("/api/chat", json={"message": "berapa grade tomat?"})
    assert r.status_code == 503
    assert "Invalid request using [API key disembunyikan]" in r.json()["detail"]
    assert secret not in r.text


def test_chat_shows_gemini_field_violations(tmp_path):
    import httpx

    def fail(**_kwargs):
        request = httpx.Request("POST", "https://example.test/generate")
        response = httpx.Response(
            400,
            request=request,
            json={
                "error": {
                    "status": "INVALID_ARGUMENT",
                    "details": [
                        {
                            "fieldViolations": [
                                {"field": "tools[0]", "description": "Unsupported function schema"}
                            ]
                        }
                    ],
                }
            },
        )
        raise httpx.HTTPStatusError("upstream detail", request=request, response=response)

    with TestClient(make_app(tmp_path, lambda: SimpleNamespace(generate_content=fail))) as c:
        r = c.post("/api/chat", json={"message": "berapa grade tomat?"})
    assert "tools[0]: Unsupported function schema" in r.json()["detail"]


def test_gemini_error_falls_back_to_raw_body_and_redacts_exact_key(tmp_path):
    import httpx

    secret = "custom-secret-key-value"

    def fail(**_kwargs):
        request = httpx.Request("POST", "https://example.test/generate")
        response = httpx.Response(400, request=request, text=f"INVALID_ARGUMENT {secret}")
        raise httpx.HTTPStatusError("upstream detail", request=request, response=response)

    client = SimpleNamespace(
        api_key=secret,
        generate_content=fail,
    )
    with TestClient(make_app(tmp_path, lambda: client)) as c:
        r = c.post("/api/chat", json={"message": "berapa grade tomat?"})
    assert "INVALID_ARGUMENT [API key disembunyikan]" in r.json()["detail"]
    assert secret not in r.text


def test_gemini_html_error_reports_proxy_diagnostic(tmp_path):
    import httpx

    def fail(**_kwargs):
        request = httpx.Request("POST", "https://example.test/generate")
        response = httpx.Response(
            400,
            request=request,
            text="<html><title>Error 400 (Bad Request)</title></html>",
            headers={"content-type": "text/html; charset=UTF-8", "server": "test-gateway"},
        )
        raise httpx.HTTPStatusError("upstream detail", request=request, response=response)

    client = SimpleNamespace(api_key="not-real", generate_content=fail)
    with TestClient(make_app(tmp_path, lambda: client)) as c:
        r = c.post("/api/chat", json={"message": "berapa grade tomat?"})
    detail = r.json()["detail"]
    assert "HTML bukan JSON" in detail
    assert "proxy/jaringan" in detail
    assert "test-gateway" in detail


def test_chat_rejects_out_of_scope_without_calling_gemini(tmp_path):
    calls = []

    def client_factory():
        calls.append(True)
        return FakeGemini()

    with TestClient(make_app(tmp_path, client_factory)) as c:
        r = c.post("/api/chat", json={"message": "Bantu saya membuat puisi tentang laut"})
        assert r.status_code == 200
        assert "hanya bisa membantu" in r.json()["reply"]
        assert len(c.get(f"/api/chat/{r.json()['session_id']}").json()) == 2
    assert calls == []


def test_chat_rate_limit_and_validation(tmp_path):
    cfg = load_config()
    cfg["dashboard"]["database"] = str(tmp_path / "t.db")
    cfg["dashboard"]["chat"]["max_per_minute"] = 1
    with TestClient(create_app(cfg, root=tmp_path, chat_client_factory=lambda: FakeGemini())) as c:
        assert c.post("/api/chat", json={"message": "grade tomat?"}).status_code == 200
        assert c.post("/api/chat", json={"message": "grade tomat?"}).status_code == 429
        assert c.post("/api/chat", json={"message": ""}).status_code == 422
        assert c.post("/api/chat", json={"message": "x" * 1001}).status_code == 422


def test_chat_tools_are_read_only_and_clamped(tmp_path):
    app = make_app(tmp_path)
    with TestClient(app) as c:
        for i in range(30):
            post_tomato(c, firmware_id=i)
        c.post("/api/ingest/events", json={"level": "ERROR", "source": "controller", "message": "kamera putus"}, headers=KEY)
    from dashboard.db import make_session_factory
    S = make_session_factory(tmp_path / "t.db")
    assert len(run_tool(S, "latest_tomatoes", {"limit": 999})["tomatoes"]) == 20  # dibatasi
    assert "image_path" not in run_tool(S, "latest_tomatoes", {})["tomatoes"][0]
    assert run_tool(S, "recent_errors", {})["errors"][0]["message"] == "kamera putus"
    assert run_tool(S, "average_latency", {})["samples"] == 30
    assert "error" in run_tool(S, "latest_tomatoes", {"limit": "1; DROP TABLE tomatoes"})
    assert "error" in run_tool(S, "execute_sql", {"query": "DELETE FROM tomatoes"})
    assert "error" in run_tool(S, "grade_summary", "bukan objek")
    assert run_tool(S, "grade_summary", {})["total"] == 30  # tabel masih utuh
    assert all(set(t["input_schema"]["properties"]) <= {"since_minutes", "limit"} for t in TOOLS)


def test_recent_errors_explains_firmware_timeout_without_claiming_root_cause(tmp_path):
    with TestClient(make_app(tmp_path)) as c:
        fw(c, type="ERR", id=2, code="GRADE_TIMEOUT")

    from dashboard.db import make_session_factory
    S = make_session_factory(tmp_path / "t.db")
    error = run_tool(S, "recent_errors", {})["errors"][0]

    assert error["message"] == "ERR GRADE_TIMEOUT pada tomat #2"
    assert "tidak menerima perintah GRADE" in error["explanation"]["meaning"]
    assert any("Controller" in cause for cause in error["explanation"]["possible_causes"])
    assert any("terminal controller" in check for check in error["explanation"]["next_checks"])
    assert "belum dapat dipastikan" in error["explanation"]["certainty"]
