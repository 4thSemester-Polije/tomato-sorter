import { useEffect, useState } from "react";
import { getTomatoes } from "../api.js";
import GradeChip from "./GradeChip.jsx";
import { dateTime, ms, pct } from "../format.js";

const FILTERS = ["", "A", "B", "C"];

export default function History({ pageSize, refreshKey, statusKey }) {
  const [grade, setGrade] = useState("");
  const [page, setPage] = useState(0);
  const [data, setData] = useState({ total: 0, rows: [] });
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    getTomatoes({ grade, limit: pageSize, offset: page * pageSize })
      .then((d) => !cancelled && (setData(d), setError(null)))
      .catch((e) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [grade, page, pageSize, refreshKey, statusKey]);

  const pages = Math.max(1, Math.ceil(data.total / pageSize));
  return (
    <section className="card wide">
      <div className="card-head">
        <h2>Riwayat</h2>
        <div className="row">
          <div className="seg" role="group" aria-label="Filter grade">
            {FILTERS.map((f) => (
              <button key={f || "all"} className={f === grade ? "active" : ""} aria-pressed={f === grade}
                onClick={() => { setGrade(f); setPage(0); }}>
                {f || "Semua"}
              </button>
            ))}
          </div>
          <a className="btn" href="/api/export.csv" download>Ekspor CSV</a>
        </div>
      </div>
      {error && <p className="error">Gagal memuat riwayat: {error}</p>}
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>#</th><th>Waktu</th><th>Grade</th><th>Kematangan</th><th>Cacat</th><th>Model</th><th>Latensi</th><th>Serial</th></tr>
          </thead>
          <tbody>
            {data.rows.length === 0 && <tr><td colSpan="8" className="muted">Belum ada data.</td></tr>}
            {data.rows.map((t) => (
              <tr key={t.id}>
                <td>{t.firmware_id ?? t.id}</td>
                <td>{dateTime(t.waktu)}</td>
                <td><GradeChip grade={t.grade} /></td>
                <td>{t.ripeness_class ? `${t.ripeness_class} ${pct(t.ripeness_confidence)}` : "-"}</td>
                <td>{t.defect_class ? `${t.defect_class} ${pct(t.defect_confidence)}` : "-"}</td>
                <td>{t.architecture ?? "-"}</td>
                <td>{ms(t.latency_ms)}</td>
                <td>{t.serial_status ?? "menunggu"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="pager">
        <button disabled={page === 0} onClick={() => setPage(page - 1)}>Sebelumnya</button>
        <span className="muted">Halaman {page + 1} dari {pages} · {data.total} tomat</span>
        <button disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}>Berikutnya</button>
      </div>
    </section>
  );
}
