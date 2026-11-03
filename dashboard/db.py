"""Model SQLAlchemy dan pembuatan database SQLite.

Tiga tabel (lihat dashboard/README.md): tomatoes, events, chat_messages.
Waktu disimpan sebagai UTC tanpa zona (naive); dikirim ke klien dengan akhiran 'Z'.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import CheckConstraint, DateTime, Float, Integer, String, Text, create_engine, event
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

GRADES = ("A", "B", "C")
SERIAL_STATUSES = ("ACK", "DONE", "ERR")
LEVELS = ("INFO", "WARN", "ERROR")
SOURCES = ("controller", "firmware", "api")
ROLES = ("user", "assistant")


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _in(column: str, values) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class Base(DeclarativeBase):
    pass


class Tomato(Base):
    """Satu tomat yang sudah dinilai model. serial_status diisi bertahap dari pesan firmware."""

    __tablename__ = "tomatoes"
    __table_args__ = (
        CheckConstraint(_in("grade", GRADES), name="ck_tomatoes_grade"),
        CheckConstraint(f"serial_status IS NULL OR {_in('serial_status', SERIAL_STATUSES)}", name="ck_tomatoes_serial"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    firmware_id: Mapped[int | None] = mapped_column(Integer, index=True)  # penghitung dari firmware; bisa mulai ulang
    waktu: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    image_path: Mapped[str | None] = mapped_column(String(300))
    ripeness_class: Mapped[str | None] = mapped_column(String(40))
    ripeness_confidence: Mapped[float | None] = mapped_column(Float)
    defect_class: Mapped[str | None] = mapped_column(String(40))
    defect_confidence: Mapped[float | None] = mapped_column(Float)
    grade: Mapped[str] = mapped_column(String(1))
    architecture: Mapped[str | None] = mapped_column(String(32))
    latency_ms: Mapped[float | None] = mapped_column(Float)
    serial_status: Mapped[str | None] = mapped_column(String(4))


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (
        CheckConstraint(_in("level", LEVELS), name="ck_events_level"),
        CheckConstraint(_in("source", SOURCES), name="ck_events_source"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    waktu: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    level: Mapped[str] = mapped_column(String(5), index=True)
    source: Mapped[str] = mapped_column(String(10))
    message: Mapped[str] = mapped_column(String(500))


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (CheckConstraint(_in("role", ROLES), name="ck_chat_role"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(9))
    content: Mapped[str] = mapped_column(Text)
    waktu: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


def make_session_factory(path) -> sessionmaker:
    """Buat file + tabel bila belum ada. Aman dipanggil berulang."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        URL.create("sqlite", database=str(path)),
        connect_args={"check_same_thread": False, "timeout": 5},
    )

    @event.listens_for(engine, "connect")
    def _pragmas(conn, _record):
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
