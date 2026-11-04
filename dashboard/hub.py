"""Penyiar data langsung: WebSocket untuk hasil/log/status, dan satu frame kamera terbaru untuk MJPEG."""
from __future__ import annotations

import asyncio
import json

from fastapi import WebSocket


class Hub:
    def __init__(self):
        self._clients: set[WebSocket] = set()
        self._frame: bytes | None = None
        self._seq = 0
        self._cond = asyncio.Condition()

    # ----- WebSocket -----
    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def broadcast(self, message: dict) -> None:
        if not self._clients:
            return
        text = json.dumps(message, ensure_ascii=False)
        for ws in list(self._clients):
            try:
                await ws.send_text(text)
            except Exception:
                self._clients.discard(ws)

    # ----- kamera -----
    @property
    def has_frame(self) -> bool:
        return self._frame is not None

    @property
    def latest_frame(self) -> bytes | None:
        return self._frame

    async def publish_frame(self, data: bytes) -> None:
        async with self._cond:
            self._frame = data
            self._seq += 1
            self._cond.notify_all()

    async def frames(self, fps: int):
        """Hasilkan frame terbaru setiap ada yang baru, paling cepat `fps` kali per detik."""
        last = -1
        interval = 1.0 / max(1, fps)
        while True:
            async with self._cond:
                try:
                    await asyncio.wait_for(self._cond.wait_for(lambda: self._frame is not None and self._seq != last), timeout=5)
                except asyncio.TimeoutError:
                    continue
                last, data = self._seq, self._frame
            yield data
            await asyncio.sleep(interval)
