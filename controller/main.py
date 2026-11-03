"""Program utama di PC. Jalankan:  python -m controller.main

Siklus per tomat (protokol v0.3, docs/PROTOCOL.md):
  1. Tunggu DETECT,<id> dari Arduino (IR terpicu, konveyor berhenti).
  2. Tunggu capture.settle_after_detect_s agar tomat diam.
  3. Ambil gambar segar dari kamera.
  4. vision.segment lalu vision.extract_features. Mask kosong berarti tidak ada tomat valid:
     PC TIDAK membalas, firmware berakhir di ERR GRADE_TIMEOUT (docs/ALUR_ALAT.md).
  5. Grade dari ml/api.py (bila ml.api_url diisi; gambar JPEG) atau ml.predict.predict(fitur).
  6. Kirim GRADE,<id>,<grade>. Pesan ACK, DONE, ERR, dan READY diproses saat tiba.
  7. Kirim hasil, pesan firmware, log, heartbeat, dan frame kamera ke dashboard (controller/dashboard_client.py).

Controller adalah SATU-SATUNYA proses yang membuka webcam dan serial. Dashboard hanya menerima kirimannya.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

from controller.config import load_config
from controller.imaging import encode_jpeg, write_jpeg_atomic

ROOT = Path(__file__).resolve().parent.parent


def _resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


class PreviewSender(threading.Thread):
    """Mengirim frame kamera terbaru ke dashboard (untuk stream MJPEG). Berbagi kamera lewat lock."""

    def __init__(self, camera, lock, client, fps, on_camera_ok=None):
        super().__init__(daemon=True)
        self._camera, self._lock, self._client = camera, lock, client
        self._interval = 1.0 / max(1, fps)
        self._on_camera_ok = on_camera_ok or (lambda ok: None)
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    frame = self._camera.read()
                jpeg = encode_jpeg(frame)
                if jpeg:
                    self._client.frame(jpeg)
                self._on_camera_ok(True)
            except Exception:
                self._on_camera_ok(False)
            self._stop_event.wait(self._interval)

    def stop(self):
        self._stop_event.set()


class Controller:
    def __init__(self, cfg, link, camera, predict_fn, client, background=None, camera_lock=None,
                 image_predict_fn=None, sleep=time.sleep, clock=time.monotonic):
        """predict_fn(fitur) -> (grade, confidence); image_predict_fn(jpeg) -> dict respons ml/api.py (diutamakan bila ada)."""
        self.cfg = cfg
        self.link = link
        self.camera = camera
        self.predict_fn = predict_fn
        self.image_predict_fn = image_predict_fn
        self.client = client
        self.background = background
        self.camera_lock = camera_lock or threading.Lock()
        self._sleep = sleep
        self._clock = clock
        self.state = "READY"
        self.state_id = None
        self.state_pesan = None
        self.kamera_ok = True
        self.photo_dir = _resolve(cfg["paths"]["debug_dir"])
        self.settle_s = cfg["capture"]["settle_after_detect_s"]
        self.mm_per_pixel = cfg["calibration"]["mm_per_pixel"]

    # ---- status ----
    def _set_state(self, state, id_tomat=None, pesan=None):
        self.state, self.state_id, self.state_pesan = state, id_tomat, pesan
        self.heartbeat()

    def heartbeat(self):
        self.client.status(self.state, self.state_id, self.state_pesan, serial_ok=True, camera_ok=self.kamera_ok)

    def set_camera_ok(self, ok):
        self.kamera_ok = bool(ok)

    # ---- satu langkah loop ----
    def step(self):
        """Baca satu pesan dari Arduino (bisa kosong saat timeout) dan tanggapi. Selalu mengirim heartbeat."""
        msg = self.link.read_message()
        if msg is None:
            self.heartbeat()
            return None
        self.client.firmware(msg)  # dashboard menurunkan status alat, status serial tomat, dan log dari sini
        kind = msg["type"]
        if kind == "READY":
            self.state, self.state_id, self.state_pesan = "READY", None, None
        elif kind == "DETECT":
            self._handle_detect(msg["id"])
        elif kind in ("ACK", "DONE", "ERR"):
            self.state, self.state_id = kind, msg["id"]
            self.state_pesan = msg["code"] if kind == "ERR" else None
        return msg

    def _handle_detect(self, id_tomat):
        t0 = self._clock()
        self.state, self.state_id, self.state_pesan = "DETECT", id_tomat, None
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
            self.client.event("WARN", "controller", f"Tomat #{id_tomat}: tidak ada objek valid di gambar, tidak dibalas")
            return

        self._set_state("GRADING", id_tomat)
        result = self._classify(id_tomat, frame, features)
        if result is None:
            return
        grade, payload = result

        payload["image_path"] = self._save_photo(id_tomat, draw_debug(frame, mask, features))
        self.link.send_grade(id_tomat, grade)
        payload.update(firmware_id=id_tomat, grade=grade, latency_ms=round((self._clock() - t0) * 1000, 1))
        self.client.tomato(payload)
        print(f"[#{id_tomat}] GRADE {grade} (merah {features['pct_red']:.0f}%, gelap {features['pct_dark']:.1f}%)")

    def _classify(self, id_tomat, frame, features):
        """Hasil: (grade, payload untuk dashboard) atau None bila gagal (GRADE tidak dikirim, firmware timeout)."""
        if self.image_predict_fn is not None:
            jpeg = encode_jpeg(frame)
            try:
                r = self.image_predict_fn(jpeg)
            except (OSError, ValueError) as e:
                print(f"[#{id_tomat}] API model gagal: {e}", file=sys.stderr)
                self.client.event("ERROR", "controller", f"Tomat #{id_tomat}: API model gagal ({e.__class__.__name__})")
                return None
            payload = {k: r.get(k) for k in ("ripeness", "defect", "architecture")}
            return r["grade"], payload
        try:
            grade, _confidence = self.predict_fn(features)  # confidence rubrik tidak punya kolom; hanya grade dikirim
        except NotImplementedError:
            print(f"[#{id_tomat}] ml/predict.py belum diimplementasikan; GRADE tidak dikirim", file=sys.stderr)
            self.client.event("ERROR", "controller", f"Tomat #{id_tomat}: model belum diimplementasikan, GRADE tidak dikirim")
            return None
        return grade, {"architecture": "rubrik"}

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
    from controller.dashboard_client import DashboardClient
    from controller.ml_client import make_image_predictor
    from controller.serial_link import SerialLink
    from ml.predict import predict

    cfg = load_config()
    validate_config(cfg)

    background = None
    bg_path = _resolve(cfg["paths"]["background_image"])
    if bg_path.is_file():
        background = cv2.imread(str(bg_path))
    else:
        print(f"Foto latar kosong tidak ditemukan ({bg_path}); memakai segmentasi saturasi (cadangan).")

    image_predict = None
    ml = cfg.get("ml") or {}
    if ml.get("api_url"):
        image_predict = make_image_predictor(ml["api_url"], ml.get("architecture"), ml.get("timeout_s", 10))
        print(f"Klasifikasi lewat API model: {ml['api_url']}")

    client = DashboardClient.from_config(cfg).start()
    cam_cfg, ser_cfg = cfg["camera"], cfg["serial"]
    camera = Camera(cam_cfg["index"], cam_cfg["width"], cam_cfg["height"], cam_cfg["flush_frames"])
    link = SerialLink(ser_cfg["port"], ser_cfg["baudrate"], ser_cfg["timeout_s"])
    time.sleep(2.0)  # banyak board reset saat port serial dibuka

    lock = threading.Lock()
    ctl = Controller(cfg, link, camera, predict, client, background=background, camera_lock=lock, image_predict_fn=image_predict)
    preview = PreviewSender(camera, lock, client, cfg["dashboard"]["stream_fps"], on_camera_ok=ctl.set_camera_ok)
    preview.start()
    client.event("INFO", "controller", "Controller mulai")
    print(f"Controller berjalan; mengirim ke {cfg['dashboard']['url']}. Tekan Ctrl+C untuk berhenti.")
    try:
        ctl.run()
    except KeyboardInterrupt:
        pass
    finally:
        preview.stop()
        client.status(serial_ok=False, camera_ok=False)
        client.event("INFO", "controller", "Controller berhenti")
        client.close()
        link.close()
        camera.release()


if __name__ == "__main__":
    main()
