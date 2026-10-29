"use strict";

// Urutan tahap mengikuti docs/PROTOCOL.md: READY -> DETECT -> (GRADING di PC) -> ACK -> DONE. ERR di luar urutan.
const STEPS = [
  { kode: "READY", label: "Siap" },
  { kode: "DETECT", label: "Tomat terdeteksi" },
  { kode: "GRADING", label: "Menghitung grade" },
  { kode: "ACK", label: "Mengantar" },
  { kode: "DONE", label: "Selesai" },
];

const INSTRUKSI = {
  READY: "Taruh satu tomat di dudukan dengan orientasi yang sama setiap kali.",
  DETECT: "Jauhkan tangan dan tunggu tomat diam.",
  GRADING: "Memotret dan menghitung grade.",
  ACK: "Grade diterima alat. Jangan taruh tomat berikutnya sebelum status Siap.",
  DONE: "Pengantaran selesai menurut waktu tempuh. Tunggu status Siap.",
  ERR: "Ambil tomat dari stasiun, catat kode error, lalu tunggu status Siap.",
};

const ARTI_GRADE = {
  A: "Matang, tanpa cacat",
  B: "Setengah matang atau cacat ringan",
  C: "Mentah atau cacat berat",
};

const HINT_ERROR = {
  GRADE_TIMEOUT: "Grade tidak sampai tepat waktu. Biasanya pemicu palsu (tangan di dudukan) atau segmentasi gagal.",
  BUSY: "Grade tiba saat alat tidak menunggu, atau ID tidak cocok.",
  BAD_CMD: "Alat tidak mengenali pesan. Periksa format protokol.",
  NO_ARRIVAL: "Tomat tidak terdeteksi sampai di wadah.",
  JAM: "Jalur tersumbat.",
};

const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
}

const fmt = (v, d = 1) => (v === null || v === undefined ? "-" : Number(v).toFixed(d));

let serverNow = Date.now() / 1000;
let pageSize = 25;
let filter = "";
let page = 0;
let lastLastRid = null;

function relTime(epoch) {
  const s = Math.max(0, Math.round(serverNow - epoch));
  if (s < 5) return "baru saja";
  if (s < 60) return `${s} detik lalu`;
  if (s < 3600) return `${Math.floor(s / 60)} menit lalu`;
  return new Date(epoch * 1000).toLocaleTimeString("id-ID", { hour: "2-digit", minute: "2-digit" });
}

function clockTime(epoch) {
  return new Date(epoch * 1000).toLocaleTimeString("id-ID", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

/* ---------- Pip koneksi ---------- */
function setPip(id, ok, text) {
  const p = $(id);
  p.className = "pip " + (ok === null ? "" : ok ? "ok" : "bad");
  p.lastChild.textContent = text;
}

/* ---------- Pita status ---------- */
function renderState(status, stale) {
  const bar = $("statebar");
  const isErr = status.state === "ERR" && !stale;
  bar.className = "statebar" + (isErr ? " err" : "") + (stale ? " stale" : "");

  const idx = STEPS.findIndex((s) => s.kode === status.state);
  const list = $("steps");
  list.replaceChildren(
    ...STEPS.map((s, i) => {
      let cls = "step";
      if (!stale && !isErr) {
        if (i < idx) cls += " done";
        if (i === idx) cls += " active";
      }
      if (isErr && s.kode === "DETECT") cls += " active"; // error terjadi setelah DETECT
      const li = el("li", { class: cls }, el("span", { class: "pill", text: s.label }));
      if (cls.includes("active")) li.setAttribute("aria-current", "step");
      return li;
    })
  );

  const ins = $("instruction");
  ins.replaceChildren();
  if (stale) {
    ins.append("Tidak ada kabar dari controller.", el("small", { text: "Jalankan controller, lalu halaman ini akan menyala sendiri." }));
  } else if (isErr) {
    ins.append(INSTRUKSI.ERR, el("small", { text: `Kode error: ${status.pesan || "tidak diketahui"}` }));
  } else {
    ins.append(INSTRUKSI[status.state] || "");
    if (status.id_tomat !== null && status.id_tomat !== undefined && status.state !== "READY") {
      ins.append(el("small", { text: `Tomat nomor ${status.id_tomat}` }));
    }
  }
}

/* ---------- Tomat terakhir ---------- */
function factsInto(dl, row, lengkap) {
  const rows = [
    ["Merah", `${fmt(row.pct_red)}% area`],
    ["Gelap", `${fmt(row.pct_dark)}% area`],
    ["Diameter", row.diameter_mm !== null ? `${fmt(row.diameter_mm)} mm` : `${fmt(row.diameter_px, 0)} px (belum dikalibrasi)`],
    ["Keyakinan", row.confidence !== null ? `${Math.round(row.confidence * 100)}%` : "-"],
    ["Latensi", row.latensi_s !== null ? `${fmt(row.latensi_s, 2)} detik` : "-"],
  ];
  if (lengkap) {
    rows.push(["Hue rata-rata", fmt(row.mean_h)], ["Saturasi rata-rata", fmt(row.mean_s)], ["Value rata-rata", fmt(row.mean_v)]);
  }
  rows.push(["Waktu", lengkap ? clockTime(row.waktu) : relTime(row.waktu)]);
  dl.replaceChildren(...rows.flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: v })]));
}

function renderLast(row) {
  $("last-empty").hidden = !!row;
  $("last-body").hidden = !row;
  if (!row) return;
  const badge = $("last-grade");
  badge.textContent = row.grade;
  badge.className = "grade-badge " + row.grade;
  $("last-title").textContent = `Tomat nomor ${row.id_tomat}, grade ${row.grade}`;
  $("last-meaning").textContent = ARTI_GRADE[row.grade];
  const photo = $("last-photo");
  if (row.foto) {
    if (lastLastRid !== row.rid) photo.src = `/api/foto/${row.rid}`;
    photo.hidden = false;
  } else {
    photo.hidden = true;
  }
  lastLastRid = row.rid;
  factsInto($("last-facts"), row, false);
}

/* ---------- Ringkasan, laju, error ---------- */
function renderSummary(stats) {
  const has = stats.total > 0;
  $("sum-empty").hidden = has;
  $("sum-body").hidden = !has;
  if (!has) return;
  $("sum-total").textContent = stats.total;
  const split = $("split");
  split.replaceChildren(
    ...["A", "B", "C"]
      .filter((g) => stats.per_grade[g] > 0)
      .map((g) => {
        const s = el("span", { class: g });
        s.style.flex = String(stats.per_grade[g]);
        return s;
      })
  );
  $("split-legend").replaceChildren(
    ...["A", "B", "C"].map((g) => {
      const pct = Math.round((100 * stats.per_grade[g]) / stats.total);
      return el(
        "li",
        {},
        el("span", { class: "letter " + g, text: g }),
        el("span", { class: "count", text: String(stats.per_grade[g]) }),
        el("span", { class: "muted", text: `${pct}%` })
      );
    })
  );
}

function renderThroughput(stats, windowMin) {
  const svg = $("spark");
  const data = stats.per_menit;
  const max = Math.max(1, ...data);
  const W = 300, H = 84, gap = 3;
  const bw = (W - gap * (data.length - 1)) / data.length;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "none");
  const ns = "http://www.w3.org/2000/svg";
  const nodes = [];
  const base = document.createElementNS(ns, "line");
  base.setAttribute("x1", 0); base.setAttribute("x2", W); base.setAttribute("y1", H - 0.5); base.setAttribute("y2", H - 0.5);
  nodes.push(base);
  data.forEach((n, i) => {
    const h = n === 0 ? 0 : Math.max(3, ((H - 6) * n) / max);
    const r = document.createElementNS(ns, "rect");
    r.setAttribute("x", i * (bw + gap));
    r.setAttribute("y", H - h);
    r.setAttribute("width", bw);
    r.setAttribute("height", h);
    r.setAttribute("rx", 1.5);
    const t = document.createElementNS(ns, "title");
    t.textContent = `${n} tomat, ${windowMin - 1 - i} menit lalu`;
    r.append(t);
    nodes.push(r);
  });
  svg.replaceChildren(...nodes);
  const sum = data.reduce((a, b) => a + b, 0);
  $("spark-caption").textContent = `${sum} tomat dalam ${windowMin} menit terakhir. Tiap batang satu menit.`;
  $("latency").textContent =
    stats.latensi_rata2_s !== null ? `Rata-rata latensi per tomat: ${fmt(stats.latensi_rata2_s, 2)} detik.` : "";
}

function renderErrors(stats) {
  const codes = Object.entries(stats.errors);
  $("err-empty").hidden = codes.length > 0;
  $("err-list").replaceChildren(
    ...codes.map(([code, n]) =>
      el(
        "li",
        {},
        el("span", { class: "code", text: code }),
        el("span", { class: "n", text: String(n) }),
        el("span", { class: "hint", text: HINT_ERROR[code] || "" })
      )
    )
  );
}

/* ---------- Kamera ---------- */
function renderFrame(frame, staleAfter) {
  const bad = !frame.ada || frame.umur_s > staleAfter;
  $("live-overlay").hidden = !bad;
}

/* ---------- Overview ---------- */
async function pollOverview() {
  try {
    const res = await fetch("/api/overview", { cache: "no-store" });
    if (!res.ok) throw new Error(res.status);
    const o = await res.json();
    serverNow = o.server_time;
    pageSize = o.pengaturan.page_size;
    setPip("pip-server", true, "Server");
    const stale = o.status.usang;
    setPip("pip-serial", stale ? null : !!o.status.serial_ok, stale ? "Alat: tanpa kabar" : o.status.serial_ok ? "Alat terhubung" : "Alat terputus");
    setPip("pip-kamera", stale ? null : !!o.status.kamera_ok, stale ? "Kamera: tanpa kabar" : o.status.kamera_ok ? "Kamera siap" : "Kamera bermasalah");
    renderState(o.status, stale);
    renderLast(o.terakhir);
    renderSummary(o.stats);
    renderThroughput(o.stats, o.pengaturan.window_min);
    renderErrors(o.stats);
    renderFrame(o.frame, o.pengaturan.stale_after_s);
  } catch (e) {
    setPip("pip-server", false, "Server terputus");
  }
}

/* ---------- Riwayat ---------- */
function chipFor(row) {
  const key = row.grade || "ERR";
  return el("span", { class: "chip " + key, text: row.grade || "Error" });
}

async function loadHistory() {
  const q = new URLSearchParams({ limit: pageSize, offset: page * pageSize });
  if (filter) q.set("hasil", filter);
  try {
    const res = await fetch("/api/history?" + q, { cache: "no-store" });
    if (!res.ok) throw new Error(res.status);
    const h = await res.json();
    const maxPage = Math.max(0, Math.ceil(h.total / pageSize) - 1);
    if (page > maxPage) { page = maxPage; return loadHistory(); }
    $("rows-empty").hidden = h.rows.length > 0;
    $("rows").replaceChildren(
      ...h.rows.map((r) => {
        const detailBtn = r.foto ? el("button", { type: "button", text: "Lihat" }) : el("span", { class: "muted", text: "-" });
        if (r.foto) detailBtn.addEventListener("click", () => openDetail(r));
        return el(
          "tr",
          {},
          el("td", { text: clockTime(r.waktu) }),
          el("td", { text: String(r.id_tomat) }),
          el("td", {}, chipFor(r), r.err_code ? el("span", { class: "muted", text: " " + r.err_code }) : ""),
          el("td", { class: "num", text: r.grade ? `${fmt(r.pct_red, 0)}%` : "-" }),
          el("td", { class: "num", text: r.grade ? `${fmt(r.pct_dark)}%` : "-" }),
          el("td", { class: "num", text: r.grade ? (r.diameter_mm !== null ? `${fmt(r.diameter_mm, 0)} mm` : `${fmt(r.diameter_px, 0)} px`) : "-" }),
          el("td", { class: "num", text: r.confidence !== null ? `${Math.round(r.confidence * 100)}%` : "-" }),
          el("td", { class: "num", text: r.latensi_s !== null ? `${fmt(r.latensi_s, 2)} s` : "-" }),
          el("td", {}, detailBtn)
        );
      })
    );
    const from = h.total === 0 ? 0 : page * pageSize + 1;
    const to = Math.min(h.total, (page + 1) * pageSize);
    $("pager-info").textContent = `${from} sampai ${to} dari ${h.total}`;
    $("prev").disabled = page === 0;
    $("next").disabled = page >= maxPage;
  } catch (e) {
    $("pager-info").textContent = "Riwayat gagal dimuat.";
  }
}

function openDetail(row) {
  $("detail-img").src = `/api/foto/${row.rid}`;
  factsInto($("detail-facts"), row, true);
  $("detail").showModal();
}

$("filters").addEventListener("click", (ev) => {
  const b = ev.target.closest("button[data-f]");
  if (!b) return;
  filter = b.dataset.f;
  page = 0;
  for (const x of $("filters").querySelectorAll("button")) x.setAttribute("aria-pressed", String(x === b));
  loadHistory();
});
$("prev").addEventListener("click", () => { page = Math.max(0, page - 1); loadHistory(); });
$("next").addEventListener("click", () => { page += 1; loadHistory(); });

// Bila stream MJPEG putus (server restart), coba sambung lagi.
$("live-img").addEventListener("error", () => setTimeout(() => { $("live-img").src = "/stream?t=" + Date.now(); }, 2000));

pollOverview();
loadHistory();
setInterval(pollOverview, 1000);
setInterval(loadHistory, 4000);
