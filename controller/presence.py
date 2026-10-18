"""Deteksi kehadiran tomat di stasiun foto (TODO: tim vision).

Rencana: bandingkan frame dengan gambar latar kosong (frame differencing).
Tomat dianggap hadir bila area perbedaan melewati ambang DAN stabil selama
presence.stable_seconds (agar bayangan tangan saat menaruh tomat tidak memicu foto).
Parameter: config.yaml -> presence.
"""


class PresenceDetector:
    def __init__(self, background_frame, diff_threshold, min_area_px, stable_seconds):
        raise NotImplementedError("TODO: tim vision")

    def update(self, frame) -> bool:
        """True bila tomat hadir dan stabil."""
        raise NotImplementedError("TODO: tim vision")
