"""Dashboard web Tomato Sorter (FastAPI).

Membaca database SQLite (dashboard/db.py) dan file frame pratinjau terbaru; tidak menyentuh serial
atau webcam, jadi bisa jalan berdampingan dengan controller. Jalankan:  python -m dashboard
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from dashboard import db

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
BOUNDARY = "frame"


def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p

def create_app(cfg: dict, root: Path = ROOT) -> FastAPI:
  
    d = cfg["dashboard"]
    db_path = _resolve(root, d["database"])
    live_frame = _resolve(root, d["live_frame"])
    data_dir = (root / "data").resolve()  # foto hanya boleh dilayani dari sini
    fps = max(1, int(d["stream_fps"]))
    stale_after = float(d["stale_after_s"])
    db.init_db(db_path)

    app = FastAPI(title="Tomato Sorter Dashboard")

    def frame_info() -> dict:
        try:
            age = time.time() - live_frame.stat().st_mtime
        except OSError:
            return {"ada": False, "umur_s": None}
        return {"ada": True, "umur_s": round(age, 1)}

    @app.get("/api/overview")
    def overview():
        now = time.time()
        status = db.get_status(db_path)
        status["usang"] = (now - status["diperbarui"]) > stale_after
        return {
            "server_time": now,
            "status": status,
            "stats": db.get_stats(db_path, window_min=int(d["throughput_window_min"]), now=now),
            "terakhir": db.get_last(db_path, graded=True),
            "error_terakhir": db.get_last(db_path, graded=False),
            "frame": frame_info(),
            "pengaturan": {
                "page_size": int(d["history_page_size"]),
                "window_min": int(d["throughput_window_min"]),
                "stale_after_s": stale_after,
            
            },
        }

    @app.get("/api/history")
    def history(
        hasil: str | None = Query(None, description="A, B, C, atau ERR"),
        limit: int = Query(25, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ):
        try:
            rows, total = db.list_history(db_path, hasil=hasil or None, limit=limit, offset=offset)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return {"total": total, "rows": rows}

    @app.get("/api/foto/{rid}")
    def foto(rid: int):
        row = db.get_row(db_path, rid)
        if not row or not row["foto"]:
            raise HTTPException(status_code=404, detail="Foto tidak ada")
        path = _resolve(root, row["foto"]).resolve()
        if not path.is_relative_to(data_dir) or not path.is_file():
            raise HTTPException(status_code=404, detail="Foto tidak ditemukan")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/export.csv")
    def export_csv():
        return PlainTextResponse(
            db.export_csv(db_path),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="tomato-sorter.csv"'},
        )

    @app.get("/api/frame")
    def frame():
        """Satu frame pratinjau terbaru (JPEG). 404 bila controller belum menulis."""
        try:
            return Response(live_frame.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})
        except OSError:
            raise HTTPException(status_code=404, detail="Belum ada frame dari kamera")

    @app.get("/stream")
    async def stream():
        """Stream MJPEG dari file frame terbaru. Frame dikirim ulang hanya bila file berubah."""

        async def gen():
            last_mtime = None
            while True:
                try:
                    mtime = live_frame.stat().st_mtime
                    if mtime != last_mtime:
                        data = live_frame.read_bytes()
                        last_mtime = mtime
                        yield (
                            f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n\r\n"
                        ).encode() + data + b"\r\n"
                except OSError:
                    pass  # file belum ada atau sedang ditimpa; coba lagi
                await asyncio.sleep(1.0 / fps)

        return StreamingResponse(
            gen(),
            media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
