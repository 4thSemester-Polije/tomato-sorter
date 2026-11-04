"""Cek perangkat di laptop ini: port serial dan indeks kamera.

Jalankan:  python -m tools.hardware_check
Port dan indeks kamera berbeda di tiap laptop. Hasilnya dipakai untuk mengisi config.local.yaml
(file lokal, tidak masuk Git), bukan config.yaml.
"""
from __future__ import annotations


def list_serial_ports() -> list[tuple[str, str]]:
    from serial.tools import list_ports

    return [(p.device, p.description) for p in sorted(list_ports.comports(), key=lambda p: p.device)]


def probe_cameras(max_index: int = 4) -> list[tuple[int, tuple[int, int]]]:
    """Indeks kamera yang bisa dibuka dan memberi frame, beserta resolusinya (lebar, tinggi)."""
    import cv2

    found = []
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index)
        try:
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    found.append((index, (frame.shape[1], frame.shape[0])))
        finally:
            cap.release()
    return found


def format_report(ports: list[tuple[str, str]], cameras: list[tuple[int, tuple[int, int]]]) -> str:
    lines = ["Port serial:"]
    lines += [f"  {device}  - {desc}" for device, desc in ports] or ["  (tidak ada; colok USB ESP32 lalu jalankan ulang)"]
    lines += ["", "Kamera:"]
    lines += [f"  indeks {i}  - {w}x{h}" for i, (w, h) in cameras] or ["  (tidak ada kamera yang bisa dibuka)"]
    lines += [
        "",
        "Isi config.local.yaml (di folder yang sama dengan config.yaml) dengan nilai laptop ini, misalnya:",
        "  serial:",
        f"    port: {ports[0][0] if ports else 'COM3'}",
        "  camera:",
        f"    index: {cameras[0][0] if cameras else 0}",
        "Catatan: webcam bawaan laptop biasanya indeks 0; webcam USB eksternal sering indeks 1.",
        "Tutup Arduino IDE Serial Monitor sebelum menjalankan controller (satu port hanya bisa dipakai satu program).",
    ]
    return "\n".join(lines)


def main() -> None:
    print(format_report(list_serial_ports(), probe_cameras()))


if __name__ == "__main__":
    main()
