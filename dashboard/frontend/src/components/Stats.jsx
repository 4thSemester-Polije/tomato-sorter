import { ms } from "../format.js";

const GRADES = ["A", "B", "C"];

export default function Stats({ stats, windowMin = 15 }) {
  if (!stats) return <section className="card"><h2>Ringkasan</h2><p className="muted">Memuat...</p></section>;
  const max = Math.max(1, ...stats.per_minute);
  const w = 100 / stats.per_minute.length;
  const rate = stats.per_minute.reduce((a, b) => a + b, 0) / stats.per_minute.length;
  return (
    <section className="card">
      <h2>Ringkasan</h2>
      <div className="totals">
        <div><span className="num">{stats.total}</span><span className="muted">tomat</span></div>
        <div><span className="num">{rate.toFixed(1)}</span><span className="muted">per menit</span></div>
        <div><span className="num">{ms(stats.avg_latency_ms)}</span><span className="muted">latensi rata-rata</span></div>
        <div><span className={`num${stats.error_count ? " warn" : ""}`}>{stats.error_count}</span><span className="muted">error</span></div>
      </div>
      <h3>Proporsi grade</h3>
      <ul className="bars">
        {GRADES.map((g) => {
          const n = stats.per_grade[g] ?? 0;
          const share = stats.total ? (n / stats.total) * 100 : 0;
          return (
            <li key={g}>
              <span className={`grade grade-${g}`}>{g}</span>
              <div className="bar"><div className={`fill fill-${g}`} style={{ width: `${share}%` }} /></div>
              <span className="bar-n">{n} <span className="muted">({share.toFixed(0)}%)</span></span>
            </li>
          );
        })}
      </ul>
      <h3>Tomat per menit ({windowMin} menit terakhir)</h3>
      <svg viewBox="0 0 100 30" className="spark" role="img" aria-label={`Tomat per menit, maksimum ${max}`} preserveAspectRatio="none">
        {stats.per_minute.map((v, i) => (
          <rect key={i} x={i * w + w * 0.1} width={w * 0.8} y={30 - (v / max) * 28} height={(v / max) * 28} className="spark-bar" />
        ))}
      </svg>
    </section>
  );
}
