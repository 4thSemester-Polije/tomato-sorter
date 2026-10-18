"""Komunikasi serial dengan Arduino memakai protokol v0.2."""
from __future__ import annotations

import time

from controller.protocol import format_grade, parse_message


class SerialLink:
    def __init__(self, port=None, baudrate=115200, timeout=1.0, serial_obj=None):
        """serial_obj boleh diisi objek tiruan (mock) untuk pengujian tanpa hardware."""
        if serial_obj is None:
            import serial  # pyserial

            serial_obj = serial.Serial(port, baudrate, timeout=timeout)
        self._ser = serial_obj

    def send_grade(self, tomato_id: int, grade: str) -> None:
        self._ser.write(format_grade(tomato_id, grade).encode("ascii"))
        self._ser.flush()

    def read_message(self) -> dict | None:
        raw = self._ser.readline()
        if not raw:
            return None
        return parse_message(raw.decode("ascii", errors="replace"))

    def wait_for(self, types=None, timeout_s: float = 10.0) -> dict | None:
        """Tunggu pesan valid bertipe tertentu (types=None: tipe apa pun). None bila timeout."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            msg = self.read_message()
            if msg is None:
                continue
            if types is None or msg["type"] in types:
                return msg
        return None

    def close(self) -> None:
        self._ser.close()
