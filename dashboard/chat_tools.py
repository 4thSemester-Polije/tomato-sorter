"""Fungsi tetap yang boleh dipanggil chatbot. SEMUANYA read-only dan berparameter terbatas.

LLM tidak pernah menulis SQL dan tidak punya jalur apa pun ke mesin.
"""
from __future__ import annotations

from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from dashboard.db import GRADES, Event, Tomato, utcnow
from dashboard.schemas import EventOut, TomatoOut

MAX_ROWS = 20
MAX_MINUTES = 7 * 24 * 60

_SINCE = {
    "type": "integer",
    "description": "Hanya hitung data dalam N menit terakhir. Kosongkan untuk semua data.",
}
_LIMIT = {"type": "integer"}

TOOLS = [
    {
        "name": "grade_summary",
        "description": "Jumlah tomat per grade (A/B/C) dan totalnya.",
        "input_schema": {"type": "object", "properties": {"since_minutes": _SINCE}},
    },
    {
        "name": "latest_tomatoes",
        "description": "Tomat terbaru (grade, kematangan, cacat, confidence, latensi, status serial).",
        "input_schema": {"type": "object", "properties": {"limit": {**_LIMIT, "description": "Default 5"}}},
    },
    {
        "name": "recent_errors",
        "description": "Log berlevel ERROR terbaru dari controller, firmware, atau API.",
        "input_schema": {"type": "object", "properties": {"limit": {**_LIMIT, "description": "Default 10"}}},
    },
    {
        "name": "average_latency",
        "description": "Rata-rata latensi (ms) dari DETECT sampai grade dikirim, dan jumlah sampelnya.",
        "input_schema": {"type": "object", "properties": {"since_minutes": _SINCE}},
    },
]


def _int(args: dict, key: str, default, lo: int, hi: int):
    value = args.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} harus bilangan bulat")
    return max(lo, min(hi, value))


def _since_filter(column, minutes):
    return [column >= utcnow() - timedelta(minutes=minutes)] if minutes else []


def grade_summary(session: Session, args: dict) -> dict:
    where = _since_filter(Tomato.waktu, _int(args, "since_minutes", None, 1, MAX_MINUTES))
    per = {g: 0 for g in GRADES}
    for g, n in session.execute(select(Tomato.grade, func.count()).where(*where).group_by(Tomato.grade)):
        per[g] = n
    return {"per_grade": per, "total": sum(per.values())}


def latest_tomatoes(session: Session, args: dict) -> dict:
    limit = _int(args, "limit", 5, 1, MAX_ROWS)
    rows = session.scalars(select(Tomato).order_by(Tomato.id.desc()).limit(limit)).all()
    return {"tomatoes": [TomatoOut.model_validate(r).model_dump(mode="json", exclude={"image_path"}) for r in rows]}


def recent_errors(session: Session, args: dict) -> dict:
    limit = _int(args, "limit", 10, 1, MAX_ROWS)
    rows = session.scalars(select(Event).where(Event.level == "ERROR").order_by(Event.id.desc()).limit(limit)).all()
    return {"errors": [EventOut.model_validate(r).model_dump(mode="json") for r in rows]}


def average_latency(session: Session, args: dict) -> dict:
    where = _since_filter(Tomato.waktu, _int(args, "since_minutes", None, 1, MAX_MINUTES))
    where.append(Tomato.latency_ms.is_not(None))
    avg, n = session.execute(select(func.avg(Tomato.latency_ms), func.count()).where(*where)).one()
    return {"avg_latency_ms": avg, "samples": n}


_REGISTRY = {f.__name__: f for f in (grade_summary, latest_tomatoes, recent_errors, average_latency)}


def run_tool(session_factory, name: str, args) -> dict:
    """Jalankan satu fungsi. Nama tak dikenal atau argumen salah menghasilkan {'error': ...}, bukan exception."""
    fn = _REGISTRY.get(name)
    if fn is None:
        return {"error": f"fungsi {name!r} tidak tersedia"}
    if not isinstance(args, dict):
        return {"error": "argumen harus objek"}
    try:
        with session_factory() as session:
            return fn(session, args)
    except ValueError as e:
        return {"error": str(e)}
