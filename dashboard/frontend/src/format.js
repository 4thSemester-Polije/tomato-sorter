export const STATE_LABEL = {
  READY: "Siap",
  DETECT: "Tomat terdeteksi",
  GRADING: "Menghitung grade",
  ACK: "Mengantar",
  DONE: "Selesai",
  ERR: "Error",
};

// Grade selalu tampil sebagai huruf; warna hanya pelengkap. Merah sengaja TIDAK dipakai untuk grade jelek,
// karena grade C juga mencakup tomat mentah.
export const GRADE_NOTE = { A: "Matang, mulus", B: "Setengah matang / cacat ringan", C: "Mentah / cacat berat" };

export const pct = (v) => (v == null ? "-" : `${Math.round(v * 100)}%`);
export const ms = (v) => (v == null ? "-" : v >= 1000 ? `${(v / 1000).toFixed(1)} dtk` : `${Math.round(v)} ms`);
export const clock = (iso) => new Date(iso).toLocaleTimeString("id-ID", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
export const dateTime = (iso) => new Date(iso).toLocaleString("id-ID", { dateStyle: "short", timeStyle: "medium" });
