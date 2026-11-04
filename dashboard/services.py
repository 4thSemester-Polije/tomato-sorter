"""Operasi database dan logika bisnis dashboard (tanpa HTTP, supaya mudah diuji)."""
from __future__ import annotations

import calendar
import csv
import io
import time
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from dashboard.db import GRADES, Event, Tomato, utcnow
from dashboard.schemas import FirmwareMessage, StatusIn, TomatoIn

MATCH_WINDOW = timedelta(minutes=10)  # ID firmware bisa mulai ulang dari 0 setelah reboot
TERMINAL = ("DONE", "ERR")


@dataclass
class DeviceState:
    state: str = "READY"
    firmware_id: int | None = None
    message: str | None = None
    serial_ok: bool = False
    camera_ok: bool = False
    updated: float = 0.0  # 0 = belum pernah ada kabar dari controller

    def to_dict(self, stale_after_s: float, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        return {
            "state": self.state,
            "firmware_id": self.firmware_id,
            "message": self.message,
            "serial_ok": self.serial_ok,
            "camera_ok": self.camera_ok,
            "updated": self.updated or None,
            "stale": self.updated == 0 or (now - self.updated) > stale_after_s,
        }


def apply_status(device: DeviceState, s: StatusIn) -> None:
    if s.state is not None:
        device.state = s.state
        device.firmware_id = s.firmware_id
        device.message = s.message
    if s.serial_ok is not None:
        device.serial_ok = s.serial_ok
    if s.camera_ok is not None:
        device.camera_ok = s.camera_ok
    device.updated = time.time()


def add_event(session: Session, level: str, source: str, message: str) -> Event:
    ev = Event(level=level, source=source, message=message[:500])
    session.add(ev)
    session.flush()
    return ev


def create_tomato(session: Session, data: TomatoIn) -> Tomato:
    t = Tomato(
        firmware_id=data.firmware_id,
        grade=data.grade,
        architecture=data.architecture,
        latency_ms=data.latency_ms,
        image_path=data.image_path,
        ripeness_class=data.ripeness.class_ if data.ripeness else None,
        ripeness_confidence=data.ripeness.confidence if data.ripeness else None,
        defect_class=data.defect.class_ if data.defect else None,
        defect_confidence=data.defect.confidence if data.defect else None,
    )
    if data.waktu is not None:
        t.waktu = data.waktu
    session.add(t)
    session.flush()
    return t


def apply_firmware_message(session: Session, device: DeviceState, msg: FirmwareMessage) -> tuple[list[Event], Tomato | None]:
    """Perbarui status alat, status serial tomat, dan log dari satu pesan protokol v0.3."""
    device.state = msg.type
    device.firmware_id = msg.id
    device.message = msg.code if msg.type == "ERR" else None
    device.serial_ok = True
    device.updated = time.time()

    events: list[Event] = []
    tomato = None
    if msg.type in ("ACK", "DONE", "ERR"):
        tomato = session.scalars(
            select(Tomato)
            .where(Tomato.firmware_id == msg.id, Tomato.waktu >= utcnow() - MATCH_WINDOW)
            .order_by(Tomato.id.desc())
            .limit(1)
        ).first()
        if tomato is not None and tomato.serial_status not in TERMINAL:
            tomato.serial_status = msg.type
        elif tomato is not None:
            tomato = None  # sudah final; tidak ada yang berubah

    if msg.type == "DETECT":
        events.append(add_event(session, "INFO", "firmware", f"Tomat #{msg.id} terdeteksi"))
    elif msg.type == "DONE":
        events.append(add_event(session, "INFO", "firmware", f"Tomat #{msg.id} selesai diantar ke jalur {msg.grade} (menurut waktu tempuh)"))
        if tomato is not None and tomato.grade != msg.grade:
            events.append(add_event(
                session, "WARN", "api",
                f"Tomat #{msg.id}: model memberi grade {tomato.grade}, firmware melapor {msg.grade}",
            ))
    elif msg.type == "ERR":
        events.append(add_event(session, "ERROR", "firmware", f"ERR {msg.code} pada tomat #{msg.id}"))
    return events, tomato


# ---------- kueri ----------
def list_tomatoes(session: Session, grade: str | None = None, limit: int = 25, offset: int = 0):
    where = []
    if grade:
        grade = grade.upper()
        if grade not in GRADES:
            raise ValueError(f"grade harus salah satu dari {GRADES}")
        where.append(Tomato.grade == grade)
    total = session.scalar(select(func.count()).select_from(Tomato).where(*where)) or 0
    rows = session.scalars(select(Tomato).where(*where).order_by(Tomato.id.desc()).limit(limit).offset(offset)).all()
    return rows, total


def list_events(session: Session, level: str | None = None, limit: int = 50):
    q = select(Event)
    if level:
        q = q.where(Event.level == level.upper())
    return session.scalars(q.order_by(Event.id.desc()).limit(limit)).all()


def compute_stats(session: Session, window_min: int = 15, now=None) -> dict:
    now = now or utcnow()
    per_grade = {g: 0 for g in GRADES}
    for g, n in session.execute(select(Tomato.grade, func.count()).group_by(Tomato.grade)):
        per_grade[g] = n
    avg = session.scalar(select(func.avg(Tomato.latency_ms)))
    errors = session.scalar(select(func.count()).select_from(Event).where(Event.level == "ERROR")) or 0

    now_min = calendar.timegm(now.timetuple()) // 60
    first = now_min - (window_min - 1)
    buckets = [0] * window_min
    since = now - timedelta(minutes=window_min + 1)
    for (w,) in session.execute(select(Tomato.waktu).where(Tomato.waktu >= since)):
        idx = calendar.timegm(w.timetuple()) // 60 - first
        if 0 <= idx < window_min:
            buckets[idx] += 1
    return {
        "total": sum(per_grade.values()),
        "per_grade": per_grade,
        "error_count": errors,
        "avg_latency_ms": avg,
        "per_minute": buckets,
    }


_CSV_COLUMNS = (
    "id", "waktu", "firmware_id", "grade", "ripeness_class", "ripeness_confidence", "defect_class",
    "defect_confidence", "architecture", "latency_ms", "serial_status", "image_path",
)


def _safe(value):
    """Cegah injeksi rumus saat CSV dibuka di spreadsheet."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@"):
        return "'" + value
    return "" if value is None else value


def export_csv(session: Session) -> str:
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(_CSV_COLUMNS)
    for t in session.scalars(select(Tomato).order_by(Tomato.id.asc())):
        row = {c: getattr(t, c) for c in _CSV_COLUMNS}
        row["waktu"] = t.waktu.isoformat(timespec="seconds") + "Z"
        w.writerow([_safe(row[c]) for c in _CSV_COLUMNS])
    return out.getvalue()
