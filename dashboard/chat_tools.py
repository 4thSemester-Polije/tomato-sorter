"""Fungsi tetap yang boleh dipanggil chatbot. SEMUANYA read-only dan berparameter terbatas.

LLM tidak pernah menulis SQL dan tidak punya jalur apa pun ke mesin.
"""
from __future__ import annotations

import re
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
        "description": (
            "Log ERROR terbaru beserta arti kode yang diketahui, kemungkinan penyebab, dan pemeriksaan berikutnya. "
            "Bedakan arti kode yang pasti dari kemungkinan penyebab."
        ),
        "input_schema": {"type": "object", "properties": {"limit": {**_LIMIT, "description": "Default 10"}}},
    },
    {
        "name": "average_latency",
        "description": "Rata-rata latensi (ms) dari DETECT sampai grade dikirim, dan jumlah sampelnya.",
        "input_schema": {"type": "object", "properties": {"since_minutes": _SINCE}},
    },
]

_ERROR_EXPLANATIONS = {
    "GRADE_TIMEOUT": {
        "meaning": "Firmware tidak menerima perintah GRADE dari controller sebelum batas waktu menunggu habis.",
        "possible_causes": [
            "Controller tidak berjalan, berhenti merespons, atau sedang memproses terlalu lama.",
            "Kamera/gambar gagal diproses atau tidak menghasilkan objek tomat yang valid.",
            "Layanan model tidak tersedia atau klasifikasi terlalu lama.",
            "Komunikasi serial USB terputus atau controller memakai port/baud rate yang salah.",
        ],
        "next_checks": [
            "Periksa terminal controller pada waktu error untuk pesan kamera, segmentasi, atau API model.",
            "Pastikan controller berjalan dan panel Kamera menerima frame.",
            "Pastikan API model aktif jika konfigurasi ml.api_url digunakan.",
            "Periksa kabel, port COM, baud rate, dan urutan pesan DETECT lalu GRADE.",
        ],
    },
    "BUSY": {
        "meaning": "Firmware menerima GRADE saat tidak sedang menunggu grade, atau ID pada GRADE berbeda dari ID tomat yang ditunggu.",
        "possible_causes": [
            "ID tomat di pesan GRADE tidak cocok dengan DETECT terakhir.",
            "Pesan GRADE terlambat atau dikirim lebih dari sekali.",
        ],
        "next_checks": [
            "Bandingkan ID pada DETECT dan GRADE di log serial.",
            "Periksa apakah firmware baru saja timeout atau menyelesaikan siklus sebelumnya.",
        ],
    },
    "BAD_CMD": {
        "meaning": "Firmware menerima perintah serial yang tidak dikenal atau formatnya tidak sesuai protokol.",
        "possible_causes": [
            "Format pesan, pemisah koma, grade, atau akhiran baris tidak sesuai.",
            "Ada teks debug/perintah dari perangkat lain pada jalur serial.",
        ],
        "next_checks": [
            "Periksa data serial yang dikirim; format grade otomatis adalah GRADE,<id>,<A|B|C>.",
            "Pastikan baud rate controller dan firmware sama.",
        ],
    },
    "JAM": {
        "meaning": "Firmware melaporkan masalah pada fase gerak dan menghentikan konveyor serta menutup gate.",
        "possible_causes": [
            "Ada hambatan mekanis atau aktuator tidak bergerak sesuai waktu yang diharapkan.",
        ],
        "next_checks": [
            "Periksa konveyor, pusher, servo, dan catu daya sebelum menjalankan ulang.",
            "Cari pesan serial sebelum ERR untuk mengetahui fase gerak yang gagal.",
        ],
    },
    "NO_ARRIVAL": {
        "meaning": "Firmware melaporkan tomat tidak tiba di tujuan.",
        "possible_causes": [
            "Tomat tersangkut atau waktu tempuh/jalur belum dikalibrasi.",
            "Kode ini masih cadangan protokol dan belum tentu didukung sensor konfirmasi.",
        ],
        "next_checks": [
            "Periksa jalur konveyor dan kalibrasi waktu tempuh.",
            "Pastikan firmware benar-benar memiliki sensor kedatangan sebelum menganggap ini terdeteksi otomatis.",
        ],
    },
}


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
    errors = []
    for row in rows:
        error = EventOut.model_validate(row).model_dump(mode="json")
        code = next(
            (name for name in _ERROR_EXPLANATIONS if re.search(rf"\b{re.escape(name)}\b", row.message)),
            None,
        )
        if code:
            explanation = {
                **_ERROR_EXPLANATIONS[code],
                "certainty": "Arti kode diketahui; penyebab spesifik belum dapat dipastikan dari log ini saja.",
            }
        else:
            explanation = {
                "meaning": f"Log mencatat error dari {row.source}: {row.message}.",
                "possible_causes": [],
                "next_checks": [
                    "Periksa log/terminal sumber error pada waktu yang sama untuk mendapatkan pesan yang lebih rinci."
                ],
                "certainty": "Penyebab spesifik tidak dapat dipastikan dari pesan log yang tersedia.",
            }
        error["explanation"] = explanation
        errors.append(error)
    return {"errors": errors}


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
