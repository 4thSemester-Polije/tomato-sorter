"""Kirim GRADE ke Arduino dan tampilkan balasan, untuk menguji protokol tanpa vision dan model.

Contoh:  python -m tools.send_grade A --port COM3
"""
import argparse
import time

from controller.config import load_config
from controller.serial_link import SerialLink


def main():
    p = argparse.ArgumentParser(description="Uji protokol v0.2: kirim GRADE dan tampilkan balasan Arduino")
    p.add_argument("grade", choices=["A", "B", "C"])
    p.add_argument("--id", type=int, default=1, help="ID tomat (default 1)")
    p.add_argument("--port", default=None, help="Port serial (default dari config.yaml)")
    p.add_argument("--wait", type=float, default=30.0, help="Batas tunggu total (detik)")
    args = p.parse_args()

    cfg = load_config()
    link = SerialLink(
        args.port or cfg["serial"]["port"],
        cfg["serial"]["baudrate"],
        cfg["serial"]["timeout_s"],
    )
    try:
        time.sleep(2.0)  # banyak board Arduino reset saat port dibuka
        link.send_grade(args.id, args.grade)
        print(f"-> GRADE,{args.id},{args.grade}")
        deadline = time.monotonic() + args.wait
        while time.monotonic() < deadline:
            msg = link.wait_for(None, timeout_s=max(0.1, deadline - time.monotonic()))
            if msg is None:
                break
            print("<-", msg)
            if msg["type"] in ("DONE", "ERR"):
                return
        print("Tidak ada DONE/ERR dalam batas waktu.")
    finally:
        link.close()


if __name__ == "__main__":
    main()
