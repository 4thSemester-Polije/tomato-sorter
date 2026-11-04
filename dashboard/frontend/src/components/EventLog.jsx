import { useState } from "react";
import { clock } from "../format.js";

const FILTERS = ["SEMUA", "WARN", "ERROR"];

export default function EventLog({ events }) {
  const [filter, setFilter] = useState("SEMUA");
  const shown = events.filter((e) => filter === "SEMUA" || (filter === "WARN" ? e.level !== "INFO" : e.level === "ERROR"));
  return (
    <section className="card">
      <div className="card-head">
        <h2>Log</h2>
        <div className="seg" role="group" aria-label="Filter log">
          {FILTERS.map((f) => (
            <button key={f} className={f === filter ? "active" : ""} onClick={() => setFilter(f)} aria-pressed={f === filter}>
              {f === "SEMUA" ? "Semua" : f === "WARN" ? "Peringatan+" : "Error"}
            </button>
          ))}
        </div>
      </div>
      {shown.length === 0 ? (
        <div className="empty">
          <span className="empty-ico" aria-hidden="true">📋</span>
          <strong>Tidak ada log</strong>
          <p>Peristiwa dari controller dan firmware akan tampil di sini.</p>
        </div>
      ) : (
        <ul className="log">
          {shown.map((e) => (
            <li key={e.id} className={`lvl-${e.level}`}>
              <span className="muted mono">{clock(e.waktu)}</span>
              <span className="lvl">{e.level}</span>
              <span className="muted">{e.source}</span>
              <span className="msg">{e.message}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}