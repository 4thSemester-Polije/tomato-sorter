"""Pengambilan gambar dari webcam (posisi tetap di atas stasiun foto)."""


class Camera:
    def __init__(self, index=0, width=640, height=480, flush_frames=5):
        import cv2  # impor di sini agar modul lain bisa diuji tanpa OpenCV

        self._cv2 = cv2
        self.flush_frames = flush_frames
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read_fresh(self):
        """Buang frame lama dari buffer, lalu ambil frame terbaru (BGR)."""
        for _ in range(self.flush_frames):
            self._cap.grab()
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("Gagal membaca frame dari webcam")
        return frame

    def release(self):
        self._cap.release()
