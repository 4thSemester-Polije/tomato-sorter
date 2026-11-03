import { useState } from "react";

/** Stream MJPEG dari /stream. Browser menampilkannya langsung lewat <img>. */
export default function CameraCard({ live }) {
  const [failed, setFailed] = useState(false);
  const [nonce, setNonce] = useState(0);
  const cameraOk = live.state?.camera_ok && !live.state?.stale;
  return (
    <section className="card camera">
      <h2>Kamera</h2>
      <div className="camera-frame">
        {cameraOk && !failed ? (
          <img key={nonce} src={`/stream?n=${nonce}`} alt="Tampilan langsung kamera stasiun foto" onError={() => setFailed(true)} />
        ) : (
          <div className="camera-empty">
            <p>{cameraOk === false || live.state?.stale ? "Belum ada gambar dari controller" : "Menunggu kamera"}</p>
            <button
              onClick={() => {
                setFailed(false);
                setNonce((n) => n + 1);
              }}
            >
              Muat ulang stream
            </button>
          </div>
        )}
      </div>
      <p className="muted small">Hanya controller yang membuka webcam; dashboard hanya menampilkan siarannya.</p>
    </section>
  );
}
