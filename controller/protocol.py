"""Format dan parsing pesan serial PC <-> Arduino. Lihat docs/PROTOCOL.md (v0.3).

Modul ini murni (tanpa pyserial) supaya mudah diuji.
Baris yang diawali '#' adalah log debug dari firmware dan diabaikan.
"""
from __future__ import annotations

VALID_GRADES = ("A", "B", "C")
ERROR_CODES = ("GRADE_TIMEOUT", "BUSY", "BAD_CMD", "NO_ARRIVAL", "JAM")


def format_grade(tomato_id: int, grade: str) -> str:
    """Bangun pesan PC -> Arduino: GRADE,<id>,<A|B|C>\\n"""
    grade = grade.upper()
    if grade not in VALID_GRADES:
        raise ValueError(f"Grade harus salah satu dari {VALID_GRADES}, bukan {grade!r}")
    if tomato_id < 0:
        raise ValueError("ID tomat harus >= 0")
    return f"GRADE,{tomato_id},{grade}\n"


def parse_message(line: str) -> dict | None:
    """Parse satu baris dari Arduino. None bila kosong, log debug, atau tidak valid."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split(",")
    kind = parts[0].upper()
    try:
        if kind == "READY" and len(parts) == 1:
            return {"type": "READY"}
        if kind == "DETECT" and len(parts) == 2:
            return {"type": "DETECT", "id": int(parts[1])}
        if kind == "ACK" and len(parts) == 2:
            return {"type": "ACK", "id": int(parts[1])}
        if kind == "DONE" and len(parts) == 3 and parts[2].upper() in VALID_GRADES:
            return {"type": "DONE", "id": int(parts[1]), "grade": parts[2].upper()}
        if kind == "ERR" and len(parts) == 3:
            return {"type": "ERR", "id": int(parts[1]), "code": parts[2].upper()}
    except ValueError:
        return None
    return None
