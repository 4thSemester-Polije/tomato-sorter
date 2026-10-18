"""Uji protokol v0.3 tanpa vision dan model.

Default: tunggu DETECT dari Arduino (taruh tomat di depan sensor IR), lalu balas GRADE dengan id yang sama.
Contoh:  python -m tools.send_grade A --port COM3
Tanpa menunggu DETECT (uji manual):  python -m tools.send_grade B --no-wait --id 5
"""
import argparse
import time

from controller.config import load_config
from controller.serial_link import SerialLink


def main():
    p = argparse.ArgumentParser(description="Uji protokol v0.3: balas DETECT dengan GRADE")
    p.add_argument("grade", choices=["A", "B", "C"])
    p.add_argument("--id", type=int, default=1, help="ID tomat (hanya untuk --no-wait)")
    p.add_argument("--port", default=None, help="Port serial (default dari config.yaml)")
    p.add_argument("--wait", type=float, default=60.0, help="Batas tunggu (detik)")
    p.add_argument("--no-wait", action="store_true", help="Kirim GRADE langsung tanpa menunggu DETECT")
    args = p.parse_args()

    cfg = load_config()
    link = SerialLink(args.port or cfg["serial"]["port"], cfg["serial"]["baudrate"], cfg["serial"]["timeout_s"])
    try:
        time.sleep(2.0)  # banyak board reset saat port dibuka
        if args.no_wait:
            tomato_id = args.id
        else:
            print("Menunggu DETECT (taruh tomat di depan sensor IR)...")
            msg = link.wait_for({"DETECT"}, timeout_s=args.wait)
            if msg is None:
                print("Tidak ada DETECT dalam batas waktu.")
                return
            print("<-", msg)
            tomato_id = msg["id"]

        link.send_grade(tomato_id, args.grade)
        print(f"-> GRADE,{tomato_id},{args.grade}")
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
