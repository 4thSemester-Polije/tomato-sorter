import GradeChip from "./GradeChip.jsx";
import { GRADE_NOTE, clock, ms, pct } from "../format.js";

export default function LastTomato({ tomato }) {
  return (
    <section className="card last-card">
      <h2>Tomat terakhir</h2>
      {!tomato ? (
        <div className="empty">
          <span className="empty-ico" aria-hidden="true">🍅</span>
          <strong>Belum ada tomat yang dinilai</strong>
          <p>Hasil tomat terakhir akan muncul di sini setelah controller mengirim data.</p>
        </div>
      ) : (
        <div className="last">
          <div className="last-photo">
            {tomato.image_path ? (
              <img src={`/api/tomatoes/${tomato.id}/image`} alt={`Foto segmentasi tomat #${tomato.firmware_id}`} />
            ) : (
              <div className="camera-empty">Tanpa foto</div>
            )}
          </div>
          <div>
            <div className="last-head">
              <GradeChip grade={tomato.grade} big />
              <div>
                <strong>Tomat #{tomato.firmware_id ?? tomato.id}</strong>
                <div className="muted">{GRADE_NOTE[tomato.grade]} · {clock(tomato.waktu)}</div>
              </div>
            </div>
            <dl className="kv">
              <dt>Kematangan</dt>
              <dd>{tomato.ripeness_class ? `${tomato.ripeness_class} (${pct(tomato.ripeness_confidence)})` : "-"}</dd>
              <dt>Cacat</dt>
              <dd>{tomato.defect_class ? `${tomato.defect_class} (${pct(tomato.defect_confidence)})` : "-"}</dd>
              <dt>Model</dt>
              <dd>{tomato.architecture ?? "-"}</dd>
              <dt>Latensi</dt>
              <dd>{ms(tomato.latency_ms)}</dd>
              <dt>Serial</dt>
              <dd>{tomato.serial_status ?? "menunggu"}</dd>
            </dl>
          </div>
        </div>
      )}
    </section>
  );
}