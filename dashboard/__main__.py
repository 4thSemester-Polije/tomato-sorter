"""Jalankan dashboard:  python -m dashboard"""
import argparse

import uvicorn

from controller.config import load_config
from dashboard.app import create_app


def main():
    ap = argparse.ArgumentParser(prog="python -m dashboard", description="Dashboard web Tomato Sorter")
    ap.add_argument("--host", help="timpa dashboard.host di config.yaml")
    ap.add_argument("--port", type=int, help="timpa dashboard.port di config.yaml")
    args = ap.parse_args()

    cfg = load_config()
    host = args.host or cfg["dashboard"]["host"]
    port = args.port or cfg["dashboard"]["port"]
    print(f"Dashboard: http://{host}:{port}")
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()