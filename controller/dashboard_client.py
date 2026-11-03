"""Klien HTTP controller -> dashboard (/api/ingest/*).

Semua pengiriman TIDAK PERNAH memblokir atau menjatuhkan loop serial: JSON masuk antrean yang dikirim thread
tersendiri (urutan terjaga), frame hanya menyimpan yang TERBARU (frame lama dibuang bila dashboard lambat).
Bila dashboard mati, controller tetap memilah tomat; pesan yang gagal dikirim dibuang dan dicatat satu baris.
"""
from __future__ import annotations

import http.client
import json
import os
import queue
import sys
import threading
import time
from urllib.parse import urlsplit


class DashboardClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout_s: float = 2.0, queue_size: int = 500):
        u = urlsplit(base_url)
        self._host, self._port, self._tls = u.hostname, u.port or (443 if u.scheme == "https" else 80), u.scheme == "https"
        self._headers = {"X-API-Key": api_key} if api_key else {}
        self._timeout = timeout_s
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._frame: bytes | None = None
        self._frame_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._conn = None
        self._last_warn = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)

    @classmethod
    def from_config(cls, cfg: dict) -> "DashboardClient":
        return cls(cfg["dashboard"]["url"], os.environ.get("DASHBOARD_INGEST_KEY"))

    # ----- API publik (tidak memblokir) -----
    def start(self) -> "DashboardClient":
        self._thread.start()
        return self

    def tomato(self, payload: dict) -> None:
        self._put("/api/ingest/tomatoes", payload)

    def firmware(self, msg: dict) -> None:
        """msg = keluaran controller.protocol.parse_message (kunci: type, id, grade, code)."""
        body = {"type": msg["type"]}
        for k_src, k_dst in (("id", "id"), ("grade", "grade"), ("code", "code")):
            if msg.get(k_src) is not None:
                body[k_dst] = msg[k_src]
        self._put("/api/ingest/firmware", body)

    def event(self, level: str, source: str, message: str) -> None:
        self._put("/api/ingest/events", {"level": level, "source": source, "message": message})

    def status(self, state=None, firmware_id=None, message=None, serial_ok=None, camera_ok=None) -> None:
        self._put("/api/ingest/status", {"state": state, "firmware_id": firmware_id, "message": message,
                                         "serial_ok": serial_ok, "camera_ok": camera_ok})

    def frame(self, jpeg: bytes) -> None:
        with self._frame_lock:
            self._frame = jpeg  # menimpa frame yang belum terkirim
        self._wake.set()

    def close(self, flush_s: float = 2.0) -> None:
        deadline = time.monotonic() + flush_s
        while (not self._q.empty() or self._frame is not None) and time.monotonic() < deadline:
            time.sleep(0.05)
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self._timeout + 1)  # biarkan pengiriman yang sedang berjalan selesai

    # ----- internal -----
    def _put(self, path: str, payload: dict) -> None:
        try:
            self._q.put_nowait((path, json.dumps(payload).encode(), "application/json"))
            self._wake.set()
        except queue.Full:
            self._warn("antrean ke dashboard penuh; pesan dibuang")

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            while True:
                try:
                    path, body, ctype = self._q.get_nowait()
                except queue.Empty:
                    break
                self._send(path, body, ctype)
            with self._frame_lock:
                frame, self._frame = self._frame, None
            if frame is not None:
                self._send("/api/ingest/frame", frame, "image/jpeg")

    def _send(self, path: str, body: bytes, ctype: str) -> None:
        try:
            if self._conn is None:
                cls = http.client.HTTPSConnection if self._tls else http.client.HTTPConnection
                self._conn = cls(self._host, self._port, timeout=self._timeout)
            self._conn.request("POST", path, body=body, headers={"Content-Type": ctype, **self._headers})
            resp = self._conn.getresponse()
            resp.read()
            if resp.status >= 400:
                self._warn(f"dashboard menolak {path}: HTTP {resp.status}")
        except (OSError, http.client.HTTPException) as e:
            self._conn = None  # sambung ulang pada pengiriman berikutnya
            self._warn(f"dashboard tidak terjangkau ({e.__class__.__name__}); data dibuang")

    def _warn(self, text: str) -> None:
        now = time.monotonic()
        if now - self._last_warn > 10:  # jangan membanjiri terminal
            self._last_warn = now
            print(f"[dashboard] {text}", file=sys.stderr)
