"""Dashboard web Tomato Sorter (FastAPI).

Prinsip: kamera hanya bisa dibuka satu proses, jadi `controller` yang memegang webcam dan MENGIRIM frame
ke sini (POST /api/ingest/frame). Dashboard menyiarkannya sebagai MJPEG (/stream). Dashboard tidak membuka
webcam maupun serial. Jalankan:  python -m dashboard
"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from dashboard import services
from dashboard.chat import ChatService, ChatUnavailable, RateLimited, RateLimiter, default_client_factory
from dashboard.db import Tomato, make_session_factory
from dashboard.hub import Hub
from dashboard.schemas import (
    ChatIn, ChatMessageOut, ChatOut, EventIn, EventOut, FirmwareMessage, Overview, StatusIn, TomatoIn, TomatoOut, TomatoPage,
)
from dashboard.security import require_ingest

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = Path(__file__).resolve().parent / "frontend" / "dist"
BOUNDARY = "frame"


def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def create_app(cfg: dict, root: Path = ROOT, chat_client_factory=default_client_factory) -> FastAPI:
    d = cfg["dashboard"]
    Session = make_session_factory(_resolve(root, d["database"]))
    data_dir = (root / "data").resolve()  # foto hanya boleh dilayani dari sini
    stale_after = float(d["stale_after_s"])
    window_min = int(d["throughput_window_min"])
    max_frame = int(d["max_frame_bytes"])
    hub = Hub()
    device = services.DeviceState()
    chat = ChatService(Session, d["chat"], chat_client_factory)
    chat_limit = RateLimiter(int(d["chat"]["max_per_minute"]))

    def state_dict() -> dict:
        return device.to_dict(stale_after)

    def snapshot() -> dict:
        with Session() as s:
            rows, _ = services.list_tomatoes(s, limit=int(d["history_page_size"]))
            return {
                "type": "snapshot",
                "state": state_dict(),
                "stats": services.compute_stats(s, window_min),
                "tomatoes": [TomatoOut.model_validate(r).model_dump(mode="json") for r in rows],
                "events": [EventOut.model_validate(e).model_dump(mode="json") for e in services.list_events(s, limit=50)],
            }

    async def push_events(events) -> None:
        for e in events:
            await hub.broadcast({"type": "event", "event": EventOut.model_validate(e).model_dump(mode="json")})

    @asynccontextmanager
    async def lifespan(_app):
        async def ticker():  # status "usang" muncul walau controller berhenti total
            while True:
                await asyncio.sleep(2)
                if hub.client_count:
                    await hub.broadcast({"type": "state", "state": state_dict()})

        task = asyncio.create_task(ticker())
        yield
        task.cancel()

    app = FastAPI(title="Tomato Sorter Dashboard", version="1.0.0", lifespan=lifespan)

    # ================= INGEST (model, controller, firmware) =================
    ingest = [Depends(require_ingest)]

    @app.post("/api/ingest/tomatoes", response_model=TomatoOut, status_code=201, dependencies=ingest, tags=["ingest"],
              summary="Hasil model untuk satu tomat")
    async def ingest_tomato(body: TomatoIn):
        with Session() as s:
            t = services.create_tomato(s, body)
            s.commit()
            out = TomatoOut.model_validate(t).model_dump(mode="json")
            stats = services.compute_stats(s, window_min)
        await hub.broadcast({"type": "tomato", "tomato": out, "stats": stats})
        return out

    @app.post("/api/ingest/firmware", dependencies=ingest, tags=["ingest"], summary="Pesan firmware (protokol v0.3)")
    async def ingest_firmware(body: FirmwareMessage):
        with Session() as s:
            events, tomato = services.apply_firmware_message(s, device, body)
            s.commit()
            ev_out = list(events)
            t_out = TomatoOut.model_validate(tomato).model_dump(mode="json") if tomato else None
            stats = services.compute_stats(s, window_min) if body.type == "ERR" else None
        await hub.broadcast({"type": "state", "state": state_dict()})
        await push_events(ev_out)
        if t_out:
            await hub.broadcast({"type": "tomato_update", "tomato": t_out})
        if stats:
            await hub.broadcast({"type": "stats", "stats": stats})
        return {"ok": True, "tomato_id": t_out["id"] if t_out else None}

    @app.post("/api/ingest/events", response_model=EventOut, status_code=201, dependencies=ingest, tags=["ingest"],
              summary="Log dari controller, firmware, atau API model")
    async def ingest_event(body: EventIn):
        with Session() as s:
            ev = services.add_event(s, body.level, body.source, body.message)
            s.commit()
            out = EventOut.model_validate(ev).model_dump(mode="json")
            stats = services.compute_stats(s, window_min) if body.level == "ERROR" else None
        await hub.broadcast({"type": "event", "event": out})
        if stats:
            await hub.broadcast({"type": "stats", "stats": stats})
        return out

    @app.post("/api/ingest/status", dependencies=ingest, tags=["ingest"], summary="Heartbeat controller")
    async def ingest_status(body: StatusIn):
        services.apply_status(device, body)
        await hub.broadcast({"type": "state", "state": state_dict()})
        return {"ok": True}

    @app.post("/api/ingest/frame", status_code=204, dependencies=ingest, tags=["ingest"],
              summary="Satu frame kamera (body = JPEG mentah)")
    async def ingest_frame(request: Request):
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_frame:
            raise HTTPException(status_code=413, detail="Frame terlalu besar")
        data = await request.body()
        if len(data) > max_frame:
            raise HTTPException(status_code=413, detail="Frame terlalu besar")
        if data[:2] != b"\xff\xd8":
            raise HTTPException(status_code=415, detail="Body harus JPEG")
        await hub.publish_frame(data)
        return Response(status_code=204)

    # ================= BACA (frontend) =================
    @app.get("/api/health", tags=["dashboard"])
    def health():
        return {"status": "ok"}

    @app.get("/api/overview", response_model=Overview, tags=["dashboard"])
    def overview():
        with Session() as s:
            rows, _ = services.list_tomatoes(s, limit=1)
            return {
                "server_time": time.time(),
                "state": state_dict(),
                "stats": services.compute_stats(s, window_min),
                "last_tomato": rows[0] if rows else None,
                "has_frame": hub.has_frame,
                "settings": {"page_size": int(d["history_page_size"]), "window_min": window_min, "stale_after_s": stale_after},
            }

    @app.get("/api/tomatoes", response_model=TomatoPage, tags=["dashboard"])
    def tomatoes(grade: str | None = Query(None, description="A, B, atau C"), limit: int = Query(25, ge=1, le=200), offset: int = Query(0, ge=0)):
        with Session() as s:
            try:
                rows, total = services.list_tomatoes(s, grade or None, limit, offset)
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))
            return {"total": total, "rows": rows}

    @app.get("/api/tomatoes/{tomato_id}", response_model=TomatoOut, tags=["dashboard"])
    def tomato(tomato_id: int):
        with Session() as s:
            t = s.get(Tomato, tomato_id)
            if t is None:
                raise HTTPException(status_code=404, detail="Tomat tidak ada")
            return t

    @app.get("/api/tomatoes/{tomato_id}/image", tags=["dashboard"])
    def tomato_image(tomato_id: int):
        with Session() as s:
            t = s.get(Tomato, tomato_id)
            rel = t.image_path if t else None
        if not rel:
            raise HTTPException(status_code=404, detail="Foto tidak ada")
        path = _resolve(root, rel).resolve()
        if not path.is_relative_to(data_dir) or not path.is_file():
            raise HTTPException(status_code=404, detail="Foto tidak ditemukan")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/events", response_model=list[EventOut], tags=["dashboard"])
    def events(level: str | None = Query(None, pattern="^(INFO|WARN|ERROR)$"), limit: int = Query(50, ge=1, le=200)):
        with Session() as s:
            return services.list_events(s, level, limit)

    @app.get("/api/export.csv", tags=["dashboard"])
    def export_csv():
        with Session() as s:
            text = services.export_csv(s)
        return PlainTextResponse(text, media_type="text/csv; charset=utf-8",
                                 headers={"Content-Disposition": 'attachment; filename="tomato-sorter.csv"'})

    @app.get("/api/frame", tags=["camera"], summary="Satu frame terbaru (JPEG)")
    def frame():
        data = hub.latest_frame
        if data is None:
            raise HTTPException(status_code=404, detail="Belum ada frame dari controller")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/stream", tags=["camera"], summary="Stream MJPEG (multipart/x-mixed-replace)")
    async def stream():
        async def gen():
            async for data in hub.frames(int(d["stream_fps"])):
                yield f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n\r\n".encode() + data + b"\r\n"

        return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
                                 headers={"Cache-Control": "no-store"})

    # ================= CHATBOT =================
    @app.post("/api/chat", response_model=ChatOut, tags=["chat"])
    def chat_post(body: ChatIn):  # def biasa: panggilan SDK memblokir, jalan di threadpool
        try:
            chat_limit.check()
        except RateLimited:
            raise HTTPException(status_code=429, detail="Terlalu banyak pertanyaan, coba lagi sebentar lagi")
        try:
            sid, reply = chat.reply(body.session_id, body.message)
        except ChatUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))
        return {"session_id": sid, "reply": reply}

    @app.get("/api/chat/{session_id}", response_model=list[ChatMessageOut], tags=["chat"])
    def chat_history(session_id: str):
        return chat.history_for(session_id)

    # ================= WEBSOCKET =================
    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await hub.connect(websocket)
        try:
            await websocket.send_text(json.dumps(snapshot(), ensure_ascii=False))
            while True:
                await websocket.receive_text()  # klien tidak mengirim apa pun; ini hanya mendeteksi putus
        except WebSocketDisconnect:
            pass
        finally:
            hub.disconnect(websocket)

    # ================= FRONTEND (hasil build Vite) =================
    if FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
    else:
        @app.get("/", include_in_schema=False)
        def no_frontend():
            return PlainTextResponse("Frontend belum dibangun. Jalankan: cd dashboard/frontend && npm install && npm run build",
                                     status_code=503)

    return app
