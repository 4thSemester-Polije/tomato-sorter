"""Program utama di PC. Jalankan:  python -m controller.main

Siklus per tomat (protokol v0.3, docs/PROTOCOL.md):
  1. Tunggu DETECT,<id> dari Arduino (IR terpicu, konveyor berhenti).
  2. Tunggu capture.settle_after_detect_s agar tomat diam.
  3. Ambil gambar segar dari kamera.
  4. vision.segment lalu vision.extract_features. Mask kosong berarti tidak ada tomat valid:
     PC TIDAK membalas, firmware berakhir di ERR GRADE_TIMEOUT (docs/ALUR_ALAT.md).
  5. ml.predict.predict(fitur) -> grade.
  6. Kirim GRADE,<id>,<grade>. Pesan ACK, DONE, ERR, dan READY diproses saat tiba.
  7. Catat hasil ke database dashboard (dashboard/db.py) dan simpan foto debug.

Pratinjau kamera untuk dashboard ditulis oleh thread terpisah ke dashboard.live_frame.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

from controller.config import load_config
from dashboard import db
from dashboard.frames import write_jpeg_atomic

ROOT = Path(__file__).resolve().parent.parent


def _resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


class PreviewWriter(threading.Thread):
    """Menulis frame kamera terbaru ke file untuk stream dashboard. Berbagi kamera lewat lock."""

    def __init__(self, camera, lock, path, fps, on_camera_ok=None):
        super().__init__(daemon=True)
        self._camera, self._lock, self._path = camera, lock, path
        self._interval = 1.0 / max(1, fps)
        self._on_camera_ok = on_camera_ok or (lambda ok: None)
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    frame = self._camera.read()
                write_jpeg_atomic(self._path, frame)
                self._on_camera_ok(True)
            except Exception:
                self._on_camera_ok(False)
            self._stop_event.wait(self._interval)

    def stop(self):
        self._stop_event.set()


class Controller:
    def __init__(self, cfg, link, camera, predict_fn, db_path, background=None, camera_lock=None, sleep=time.sleep, clock=time.monotonic):
        self.cfg = cfg
        self.link = link
        self.camera = camera
        self.predict_fn = predict_fn
        self.db_path = db_path
        self.background = background
        self.camera_lock = camera_lock or threading.Lock()
        self._sleep = sleep
        self._clock = clock
        self.state = "READY"
        self.state_id = None
        self.state_pesan = None
        self.kamera_ok = True
        self.pending = {}  # id_tomat -> hasil yang menunggu DONE
        self.photo_dir = _resolve(cfg["paths"]["debug_dir"])
        self.settle_s = cfg["capture"]["settle_after_detect_s"]
        self.mm_per_pixel = cfg["calibration"]["mm_per_pixel"]

    # ---- status ----
    def _set_state(self, state, id_tomat=None, pesan=None):
        self.state, self.state_id, self.state_pesan = state, id_tomat, pesan
        self.heartbeat()

    def heartbeat(self):
        db.set_status(
            self.db_path, self.state, id_tomat=self.state_id, pesan=self.state_pesan,
            serial_ok=True, kamera_ok=self.kamera_ok,
        )

    def set_camera_ok(self, ok):
        self.kamera_ok = bool(ok)

    # ---- satu langkah loop ----
    def step(self):
        """Baca satu pesan dari Arduino (bisa kosong saat timeout) dan tanggapi. Selalu mengirim heartbeat."""
        msg = self.link.read_message()
        if msg is None:
            self.heartbeat()
            return None
        kind = msg["type"]
        if kind == "READY":
            self._set_state("READY")
        elif kind == "DETECT":
            self._handle_detect(msg["id"])
        elif kind == "ACK":
            self._set_state("ACK", msg["id"])
        elif kind == "DONE":
            self._handle_done(msg)
        elif kind == "ERR":
            self._handle_err(msg)
        return msg

    def _handle_detect(self, id_tomat):
        t0 = self._clock()
        self._set_state("DETECT", id_tomat)
        self._sleep(self.settle_s)
        with self.camera_lock:
            frame = self.camera.read_fresh()

        from vision.debug import draw_debug
        from vision.features import extract_features
        from vision.segmentation import segment

        mask = segment(frame, self.background)
        try:
            features = extract_features(frame, mask, self.mm_per_pixel)
        except ValueError:
            print(f"[#{id_tomat}] tidak ada tomat valid di gambar; tidak membalas, menunggu GRADE_TIMEOUT dari firmware")
            return

        self._set_state("GRADING", id_tomat)
        try:
            grade, confidence = self.predict_fn(features)
        except NotImplementedError:
            print(f"[#{id_tomat}] ml/predict.py belum diimplementasikan; GRADE tidak dikirim", file=sys.stderr)
            return

        foto = self._save_photo(id_tomat, draw_debug(frame, mask, features))
        self.link.send_grade(id_tomat, grade)
        self.pending[id_tomat] = {
            "features": features, "grade": grade, "confidence": confidence,
            "latensi_s": self._clock() - t0, "foto": foto,
        }
        print(f"[#{id_tomat}] GRADE {grade} (merah {features['pct_red']:.0f}%, gelap {features['pct_dark']:.1f}%)")

    def _handle_done(self, msg):
        self._set_state("DONE", msg["id"])
        res = self.pending.pop(msg["id"], None)
        if res is None:
            return  # DONE tanpa GRADE dari PC, misalnya uji manual lewat serial monitor
        db.insert_result(self.db_path, msg["id"], **res)

    def _handle_err(self, msg):
        self._set_state("ERR", msg["id"], pesan=msg["code"])
        self.pending.pop(msg["id"], None)
        try:
            db.insert_result(self.db_path, msg["id"], err_code=msg["code"])
        except ValueError:
            print(f"Kode error tidak dikenal dari firmware: {msg['code']}", file=sys.stderr)

    def _save_photo(self, id_tomat, image):
        path = self.photo_dir / f"T{id_tomat:03d}_{int(time.time() * 1000)}.jpg"
        write_jpeg_atomic(path, image)
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            return str(path)

    def run(self, stop_event=None):
        stop_event = stop_event or threading.Event()
        while not stop_event.is_set():
            self.step()


def validate_config(cfg):
    if cfg["capture"]["settle_after_detect_s"] is None:
        raise SystemExit(
            "capture.settle_after_detect_s di config.yaml masih null. Ukur jeda tomat diam setelah DETECT, "
            "lalu isi nilainya (untuk uji awal boleh 1.0)."
        )


def main():
    import cv2

    from controller.camera import Camera
    from controller.serial_link import SerialLink
    from ml.predict import predict

    cfg = load_config()
    validate_config(cfg)
    d = cfg["dashboard"]
    db_path = _resolve(d["database"])
    db.init_db(db_path)

    background = None
    bg_path = _resolve(cfg["paths"]["background_image"])
    if bg_path.is_file():
        background = cv2.imread(str(bg_path))
    else:
        print(f"Foto latar kosong tidak ditemukan ({bg_path}); memakai segmentasi saturasi (cadangan).")

    cam_cfg, ser_cfg = cfg["camera"], cfg["serial"]
    camera = Camera(cam_cfg["index"], cam_cfg["width"], cam_cfg["height"], cam_cfg["flush_frames"])
    link = SerialLink(ser_cfg["port"], ser_cfg["baudrate"], ser_cfg["timeout_s"])
    time.sleep(2.0)  # banyak board reset saat port serial dibuka

    lock = threading.Lock()
    ctl = Controller(cfg, link, camera, predict, db_path, background=background, camera_lock=lock)
    preview = PreviewWriter(camera, lock, _resolve(d["live_frame"]), d["stream_fps"], on_camera_ok=ctl.set_camera_ok)
    preview.start()
    print(f"Controller berjalan. Database: {db_path}. Tekan Ctrl+C untuk berhenti.")
    try:
        ctl.run()
    except KeyboardInterrupt:
        pass
    finally:
        preview.stop()
        db.set_status(db_path, ctl.state, serial_ok=False, kamera_ok=False)
        link.close()
        camera.release()


if __name__ == "__main__":
    main()