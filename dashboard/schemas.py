"""Kontrak data API (dibuat otomatis menjadi OpenAPI di /openapi.json dan docs/openapi.json)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, field_validator, model_validator

Grade = Literal["A", "B", "C"]
UtcDatetime = Annotated[datetime, PlainSerializer(lambda d: d.isoformat(timespec="seconds") + "Z", return_type=str)]


def _to_naive_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


# ---------- masukan dari model / controller / firmware ----------
class ClassResult(BaseModel):
    """Bentuk sama dengan `ripeness` / `defect` pada respons POST /predict di ml/api.py (field lain diabaikan)."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    class_: str = Field(alias="class", min_length=1, max_length=40)
    confidence: float = Field(ge=0, le=1)


class TomatoIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    firmware_id: int | None = Field(None, ge=0, description="ID dari firmware (DETECT,<id>)")
    ripeness: ClassResult | None = None
    defect: ClassResult | None = None
    grade: Grade
    architecture: str | None = Field(None, max_length=32)
    latency_ms: float | None = Field(None, ge=0)
    image_path: str | None = Field(None, max_length=300, description="Relatif terhadap akar proyek, harus di dalam data/")
    waktu: datetime | None = Field(None, description="Bawaan: waktu server")

    _utc = field_validator("waktu")(_to_naive_utc)


class FirmwareMessage(BaseModel):
    """Pesan firmware apa adanya, sama dengan keluaran controller.protocol.parse_message."""

    type: Literal["READY", "DETECT", "ACK", "DONE", "ERR"]
    id: int | None = Field(None, ge=0)
    grade: Grade | None = None
    code: str | None = Field(None, pattern=r"^[A-Z_]{1,32}$")

    @model_validator(mode="after")
    def _check(self):
        if self.type != "READY" and self.id is None:
            raise ValueError(f"{self.type} wajib punya id")
        if self.type == "DONE" and self.grade is None:
            raise ValueError("DONE wajib punya grade")
        if self.type == "ERR" and self.code is None:
            raise ValueError("ERR wajib punya code")
        return self


class EventIn(BaseModel):
    level: Literal["INFO", "WARN", "ERROR"]
    source: Literal["controller", "firmware", "api"]
    message: str = Field(min_length=1, max_length=500)


class StatusIn(BaseModel):
    """Heartbeat controller. Semua field opsional; yang kosong tidak mengubah nilai sebelumnya."""

    state: Literal["READY", "DETECT", "GRADING", "ACK", "DONE", "ERR"] | None = None
    firmware_id: int | None = Field(None, ge=0)
    message: str | None = Field(None, max_length=200)
    serial_ok: bool | None = None
    camera_ok: bool | None = None


class ChatIn(BaseModel):
    session_id: str | None = Field(None, pattern=r"^[A-Za-z0-9_-]{8,64}$")
    message: str = Field(min_length=1, max_length=1000)


# ---------- keluaran ----------
class TomatoOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    firmware_id: int | None
    waktu: UtcDatetime
    image_path: str | None
    ripeness_class: str | None
    ripeness_confidence: float | None
    defect_class: str | None
    defect_confidence: float | None
    grade: Grade
    architecture: str | None
    latency_ms: float | None
    serial_status: Literal["ACK", "DONE", "ERR"] | None


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    waktu: UtcDatetime
    level: Literal["INFO", "WARN", "ERROR"]
    source: Literal["controller", "firmware", "api"]
    message: str


class ChatMessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    role: Literal["user", "assistant"]
    content: str
    waktu: UtcDatetime


class ChatOut(BaseModel):
    session_id: str
    reply: str


class DeviceStateOut(BaseModel):
    state: str
    firmware_id: int | None
    message: str | None
    serial_ok: bool
    camera_ok: bool
    updated: float | None = Field(description="Epoch detik pesan terakhir dari controller; null bila belum pernah")
    stale: bool = Field(description="True bila controller tidak memberi kabar lebih lama dari dashboard.stale_after_s")


class StatsOut(BaseModel):
    total: int
    per_grade: dict[str, int]
    error_count: int
    avg_latency_ms: float | None
    per_minute: list[int] = Field(description="Tomat per menit, terlama dulu; elemen terakhir = menit berjalan")


class TomatoPage(BaseModel):
    total: int
    rows: list[TomatoOut]


class Overview(BaseModel):
    server_time: float
    state: DeviceStateOut
    stats: StatsOut
    last_tomato: TomatoOut | None
    has_frame: bool
    settings: dict
