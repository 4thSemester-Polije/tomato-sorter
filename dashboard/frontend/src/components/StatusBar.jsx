import { STATE_LABEL } from "../format.js";

export default function StatusBar({ live }) {
  const { connected, state } = live;
  let tone = "ok";
  let title = "Menunggu data";
  let sub = "";
  if (!connected) {
    tone = "off";
    title = "Dashboard terputus";
    sub = "Mencoba menyambung ulang...";
  } else if (!state || state.stale) {
    tone = "off";
    title = "Controller tidak aktif";
    sub = "Jalankan: python -m controller.main";
  } else {
    title = STATE_LABEL[state.state] ?? state.state;
    if (state.state === "ERR") tone = "err";
    else if (state.state === "READY") tone = "ok";
    else tone = "busy";
    sub = [
      state.firmware_id != null && `Tomat #${state.firmware_id}`,
      state.message,
    ]
      .filter(Boolean)
      .join(" · ");
    if (state.state === "DONE")
      sub += (sub ? " · " : "") + "selesai menurut waktu tempuh";
  }
  return (
    <header
      className={`status status-${tone}`}
      role="status"
      aria-live="polite"
    >
      <div className="brand">
        <span className="logo" aria-hidden="true">
          🍅
        </span>
        <div>
          <h1>Tomato Sorter</h1>
          <p className="muted">SMART SORTING · LIVE MONITOR</p>
        </div>
      </div>
      <div className="status-main">
        <span className="dot" />
        <div>
          <strong>{title}</strong>
          {sub && <span className="status-sub">{sub}</span>}
        </div>
      </div>
      <ul className="pills">
        <li className={state?.serial_ok && !state?.stale ? "pill on" : "pill"}>
          Serial {state?.serial_ok && !state?.stale ? "OK" : "-"}
        </li>
        <li className={state?.camera_ok && !state?.stale ? "pill on" : "pill"}>
          Kamera {state?.camera_ok && !state?.stale ? "OK" : "-"}
        </li>
      </ul>
    </header>
  );
}
