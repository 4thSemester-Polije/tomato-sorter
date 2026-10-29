"""Penyimpanan hasil sortir dan status alat (SQLite).

Jembatan antara controller (menulis) dan dashboard (membaca). Keduanya proses terpisah,
jadi dashboard tidak perlu tahu apa pun tentang serial atau kamera.

Pemakaian dari controller (lihat dashboard/README.md):

    from dashboard import db
    db.init_db(path)
    db.set_status(path, "READY", serial_ok=True, kamera_ok=True)   # ulangi tiap <= 2 detik (heartbeat)
    db.insert_result(path, 12, features=fitur, grade="B", confidence=0.87, latensi_s=1.2, foto="data/debug/T012.jpg")
    db.insert_result(path, 13, err_code="GRADE_TIMEOUT")
"""
from __future__ import annotations

import csv
import io
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from controller.protocol import ERROR_CODES, VALID_GRADES

STATES = ("READY", "DETECT", "GRADING", "ACK", "DONE", "ERR")
GRADES = VALID_GRADES

# Harus sama dengan vision.features.FEATURE_NAMES (kontrak antar modul; dijaga oleh tests/test_dashboard.py).
FEATURE_COLUMNS = ("pct_red", "mean_h", "mean_s", "mean_v", "pct_dark", "diameter_px", "diameter_mm")

CSV_COLUMNS = ("waktu", "id_tomat", "grade", "confidence", "latensi_s", "err_code") + FEATURE_COLUMNS + ("foto",)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hasil (
    rid         INTEGER PRIMARY KEY AUTOINCREMENT,
    id_tomat    INTEGER NOT NULL,
    waktu       INTEGER NOT NULL,
    pct_red     REAL, mean_h REAL, mean_s REAL, mean_v REAL,
    pct_dark    REAL, diameter_px REAL, diameter_mm REAL,
    grade       TEXT CHECK (grade IS NULL OR grade IN ('A','B','C')),
    confidence  REAL,
    latensi_s   REAL,
    err_code    TEXT,
    foto        TEXT
);
CREATE INDEX IF NOT EXISTS idx_hasil_waktu ON hasil (waktu);
CREATE TABLE IF NOT EXISTS status (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    state       TEXT NOT NULL,
    id_tomat    INTEGER,
    pesan       TEXT,
    serial_ok   INTEGER NOT NULL DEFAULT 0,
    kamera_ok   INTEGER NOT NULL DEFAULT 0,
    diperbarui  INTEGER NOT NULL
);
"""


@contextmanager
def _connect(path):
    conn = sqlite3.connect(str(path), timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path) -> None:
    """Buat file dan tabel bila belum ada. Aman dipanggil berulang."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as c:
        c.execute("PRAGMA journal_mode=WAL")  # controller menulis, dashboard membaca, tanpa saling mengunci
        c.executescript(_SCHEMA)
        c.execute(
            "INSERT OR IGNORE INTO status (id, state, diperbarui) VALUES (1, 'READY', 0)"
        )  # diperbarui=0: belum pernah ada kabar dari controller


def set_status(path, state, id_tomat=None, pesan=None, serial_ok=None, kamera_ok=None, now=None) -> None:
    """Perbarui status alat. serial_ok / kamera_ok = None berarti pertahankan nilai sebelumnya."""
    if state not in STATES:
        raise ValueError(f"state harus salah satu dari {STATES}, bukan {state!r}")
    now = int(time.time() if now is None else now)
    with _connect(path) as c:
        old = c.execute("SELECT serial_ok, kamera_ok FROM status WHERE id = 1").fetchone()
        s_ok = int(bool(serial_ok)) if serial_ok is not None else (old["serial_ok"] if old else 0)
        k_ok = int(bool(kamera_ok)) if kamera_ok is not None else (old["kamera_ok"] if old else 0)
        c.execute(
            "INSERT INTO status (id, state, id_tomat, pesan, serial_ok, kamera_ok, diperbarui) "
            "VALUES (1, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET state=excluded.state, id_tomat=excluded.id_tomat, pesan=excluded.pesan, "
            "serial_ok=excluded.serial_ok, kamera_ok=excluded.kamera_ok, diperbarui=excluded.diperbarui",
            (state, id_tomat, pesan, s_ok, k_ok, now),
        )


def get_status(path) -> dict:
    with _connect(path) as c:
        row = c.execute("SELECT * FROM status WHERE id = 1").fetchone()
    if row is None:
        return {"state": "READY", "id_tomat": None, "pesan": None, "serial_ok": 0, "kamera_ok": 0, "diperbarui": 0}
    d = dict(row)
    d.pop("id", None)
    return d


def insert_result(
    path,
    id_tomat,
    features=None,
    grade=None,
    confidence=None,
    latensi_s=None,
    err_code=None,
    foto=None,
    waktu=None,
) -> int:
    """Simpan satu siklus tomat. Beri `grade` (berhasil) ATAU `err_code` (gagal). Mengembalikan rid."""
    if (grade is None) == (err_code is None):
        raise ValueError("Isi tepat salah satu: grade atau err_code")
    if grade is not None:
        grade = grade.upper()
        if grade not in GRADES:
            raise ValueError(f"Grade harus salah satu dari {GRADES}, bukan {grade!r}")
    if err_code is not None:
        err_code = err_code.upper()
        if err_code not in ERROR_CODES:
            raise ValueError(f"Kode error harus salah satu dari {ERROR_CODES}, bukan {err_code!r}")
    features = features or {}
    unknown = set(features) - set(FEATURE_COLUMNS)
    if unknown:
        raise ValueError(f"Fitur tidak dikenal: {sorted(unknown)}")
    waktu = int(time.time() if waktu is None else waktu)
    cols = ["id_tomat", "waktu", "grade", "confidence", "latensi_s", "err_code", "foto"] + list(FEATURE_COLUMNS)
    vals = [id_tomat, waktu, grade, confidence, latensi_s, err_code, foto] + [features.get(k) for k in FEATURE_COLUMNS]
    with _connect(path) as c:
        cur = c.execute(
            f"INSERT INTO hasil ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", vals
        )
        return cur.lastrowid

def get_row(path, rid):
    with _connect(path) as c:
        row = c.execute("SELECT * FROM hasil WHERE rid = ?", (rid,)).fetchone()
    return dict(row) if row else None


def get_last(path, graded=True):
    """Baris terbaru. graded=True: hanya yang punya grade. graded=False: hanya yang error."""
    cond = "grade IS NOT NULL" if graded else "err_code IS NOT NULL"
    with _connect(path) as c:
        row = c.execute(f"SELECT * FROM hasil WHERE {cond} ORDER BY rid DESC LIMIT 1").fetchone()
    return dict(row) if row else None


def get_stats(path, window_min=15, now=None) -> dict:
    """Ringkasan: jumlah per grade, error per kode, rata-rata latensi, dan tomat per menit."""
    now = int(time.time() if now is None else now)
    with _connect(path) as c:
        per_grade = {g: 0 for g in GRADES}
        for r in c.execute("SELECT grade, COUNT(*) AS n FROM hasil WHERE grade IS NOT NULL GROUP BY grade"):
            per_grade[r["grade"]] = r["n"]
        errors = {r["err_code"]: r["n"] for r in c.execute(
            "SELECT err_code, COUNT(*) AS n FROM hasil WHERE err_code IS NOT NULL GROUP BY err_code ORDER BY n DESC"
        )}
        avg = c.execute("SELECT AVG(latensi_s) AS a FROM hasil WHERE grade IS NOT NULL AND latensi_s IS NOT NULL").fetchone()["a"]
        first_minute = now // 60 - (window_min - 1)
        buckets = [0] * window_min
        for r in c.execute(
            "SELECT waktu / 60 AS menit, COUNT(*) AS n FROM hasil WHERE grade IS NOT NULL AND waktu / 60 >= ? "
            "GROUP BY menit",
            (first_minute,),
        ):
            idx = r["menit"] - first_minute
            if 0 <= idx < window_min:
                buckets[idx] = r["n"]
    return {
        "total": sum(per_grade.values()),
        "per_grade": per_grade,
        "errors": errors,
        "error_total": sum(errors.values()),
        "latensi_rata2_s": avg,
        "per_menit": buckets,
    }


def list_history(path, hasil=None, limit=25, offset=0):
    """Riwayat terbaru dulu. hasil: None (semua), 'A'/'B'/'C', atau 'ERR'. Mengembalikan (baris, total)."""
    where, params = "", []
    if hasil:
        hasil = hasil.upper()
        if hasil == "ERR":
            where = "WHERE err_code IS NOT NULL"
        elif hasil in GRADES:
            where, params = "WHERE grade = ?", [hasil]
        else:
            raise ValueError(f"Filter hasil harus A, B, C, atau ERR, bukan {hasil!r}")
    with _connect(path) as c:
        total = c.execute(f"SELECT COUNT(*) AS n FROM hasil {where}", params).fetchone()["n"]
        rows = c.execute(
            f"SELECT * FROM hasil {where} ORDER BY rid DESC LIMIT ? OFFSET ?", params + [limit, offset]
        ).fetchall()
    return [dict(r) for r in rows], total


def export_csv(path) -> str:
    """Seluruh riwayat sebagai teks CSV (urut waktu naik). Waktu memakai zona waktu PC."""
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    with _connect(path) as c:
        for r in c.execute("SELECT * FROM hasil ORDER BY rid ASC"):
            row = dict(r)
            row["waktu"] = datetime.fromtimestamp(row["waktu"]).isoformat(timespec="seconds")
            w.writerow(["" if row[k] is None else row[k] for k in CSV_COLUMNS])
    return out.getvalue()
