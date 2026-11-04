# dashboard/

Dashboard web: status alat, kamera langsung, hasil grade, log, riwayat, dan chatbot read-only.

| Bagian | Pilihan |
|---|---|
| Backend | FastAPI (`dashboard/`), satu stack dengan `ml/api.py` |
| Database | SQLite + SQLAlchemy (`data/tomato.db`): tabel `tomatoes`, `events`, `chat_messages` |
| Kamera | MJPEG `GET /stream` (`multipart/x-mixed-replace`) |
| Data langsung | WebSocket `/ws` (hasil, log, status, statistik) |
| Frontend | React + Vite (`dashboard/frontend/`), kontrak lewat `docs/openapi.json` |
| Chatbot | Gemini API dari backend, tool calling ke 4 fungsi read-only |

## Prinsip arsitektur

Kamera hanya bisa dibuka satu proses, jadi **controller** yang memegang webcam dan serial, lalu MENGIRIM
semuanya ke dashboard lewat HTTP (`controller/dashboard_client.py`). Dashboard tidak membuka webcam maupun serial.

```
webcam ─┐                     ┌─ POST /api/ingest/frame ───────► /stream (MJPEG) ─┐
serial ─┼─► controller ───────┼─ POST /api/ingest/firmware|status|events ─► DB ───┼─► browser (React)
ml/api ─┘   (python -m        └─ POST /api/ingest/tomatoes ────► DB + /ws ────────┘
            controller.main)
```

Bila dashboard mati, controller tetap memilah tomat (pengiriman tidak pernah memblokir loop serial; data yang
gagal terkirim dibuang dan dicatat satu baris di terminal).

## Menjalankan

```
pip install -r requirements.txt
cd dashboard/frontend && npm install && npm run build && cd ../..    # sekali, atau setiap UI berubah
python -m dashboard                    # terminal 1 -> http://127.0.0.1:8000
python -m controller.main              # terminal 2
uvicorn ml.api:app --port 9000         # opsional, lalu isi ml.api_url di config.yaml
```

Mode pengembangan UI: jalankan `python -m dashboard`, lalu `npm run dev` di `dashboard/frontend`
(proxy ke port 8000, hot reload).

Environment (hanya di server, tidak pernah di frontend):

| Variabel | Fungsi |
|---|---|
| `GEMINI_API_KEY` | Mengaktifkan chatbot. Buat API key di Google AI Studio. Tanpa ini, pertanyaan dalam topik chatbot menjawab 503; bagian lain tetap berjalan normal. |
| `DASHBOARD_INGEST_KEY` | Kunci untuk `/api/ingest/*` (header `X-API-Key`). Controller membaca variabel yang sama. Bila kosong, ingest hanya diterima dari mesin yang sama (loopback). |

Gemini API menyediakan free tier dengan batas kuota dan ketersediaan yang dapat berbeda menurut model, wilayah, dan kebijakan Google. Periksa ketentuan terbaru di Google AI Studio; penggunaan gratis tidak dijamin tanpa batas. Kunci API hanya disimpan di environment backend dan tidak boleh dimasukkan ke frontend atau Git.

Buat API key melalui [Google AI Studio](https://aistudio.google.com/app/apikey), lalu set di PowerShell yang sama dengan terminal dashboard:

```
$env:GEMINI_API_KEY = "API_KEY_ANDA"
python -m dashboard
```

Variabel tersebut hanya berlaku untuk sesi terminal itu. Jangan menaruh API key di `config.yaml`, kode frontend, atau commit.

Jika chat menampilkan error, periksa pesan yang lebih spesifik di dashboard dan terminal server. Error 401/403 biasanya
terkait API key atau izin project; 404 berarti model tidak tersedia; 429 berarti kuota atau batas permintaan tercapai.
Jangan pernah membagikan API key saat meminta bantuan.

Semua pengaturan lain ada di `config.yaml` bagian `dashboard` dan `ml`.

## API untuk model, controller, dan firmware (`/api/ingest/*`)

Butuh `X-API-Key` (atau loopback, lihat di atas). Skema lengkap: `/docs` atau `docs/openapi.json`
(`python -m dashboard.export_openapi`).

| Endpoint | Pengirim | Isi |
|---|---|---|
| `POST /api/ingest/tomatoes` | model / controller | Hasil satu tomat. Menerima bentuk respons `POST /predict` di `ml/api.py` (`ripeness`, `defect`, `grade`, `architecture`; field lain diabaikan) ditambah `firmware_id`, `latency_ms`, `image_path` |
| `POST /api/ingest/firmware` | firmware (lewat controller) | Pesan protokol v0.3 apa adanya: `{"type":"DONE","id":12,"grade":"B"}`, `ERR` butuh `code` |
| `POST /api/ingest/events` | siapa pun | `{"level":"INFO\|WARN\|ERROR","source":"controller\|firmware\|api","message":"..."}` |
| `POST /api/ingest/status` | controller | Heartbeat: `state`, `firmware_id`, `message`, `serial_ok`, `camera_ok` |
| `POST /api/ingest/frame` | controller | Body = JPEG mentah (maks `dashboard.max_frame_bytes`) |

Contoh dari model (mis. skrip lain atau ESP32 dengan Wi-Fi):

```
curl -X POST http://127.0.0.1:8000/api/ingest/tomatoes -H "X-API-Key: $DASHBOARD_INGEST_KEY" \
  -H "Content-Type: application/json" \
  -d '{"firmware_id":12,"grade":"B","architecture":"resnext101","latency_ms":850,
       "ripeness":{"class":"setengah","confidence":0.81},"defect":{"class":"mulus","confidence":0.9}}'
```

Perilaku di sisi dashboard saat pesan firmware masuk:

| Pesan | Status alat | Efek lain |
|---|---|---|
| `READY` | Siap | - |
| `DETECT,<id>` | Tomat terdeteksi | log INFO |
| `ACK,<id>` | Mengantar | `serial_status=ACK` pada tomat terbaru ber-`firmware_id` itu (10 menit terakhir; ID firmware bisa mulai ulang) |
| `DONE,<id>,<g>` | Selesai | `serial_status=DONE`; log WARN bila grade firmware beda dari grade model |
| `ERR,<id>,<kode>` | Error | `serial_status=ERR`, log ERROR |

Status `DONE` ditampilkan sebagai "selesai menurut waktu tempuh", bukan "terkonfirmasi" (lihat `docs/PROTOCOL.md`).

## API untuk frontend

`GET /api/overview`, `/api/tomatoes` (filter `grade`, paging), `/api/tomatoes/{id}`, `/api/tomatoes/{id}/image`,
`/api/events`, `/api/export.csv`, `/api/frame`, `/stream`, `POST /api/chat`, `GET /api/chat/{session}`, WebSocket `/ws`.

Pesan WebSocket: `snapshot` (saat tersambung), `state`, `stats`, `tomato`, `tomato_update`, `event`.

## Chatbot

Gemini dipanggil dari backend dengan tool calling ke empat fungsi tetap di `dashboard/chat_tools.py`:
`grade_summary`, `latest_tomatoes`, `recent_errors`, `average_latency`. Semuanya read-only dengan parameter
terbatas (jumlah baris maks 20). LLM tidak menulis SQL dan tidak punya jalur ke mesin. Teks log yang dibaca
chatbot diperlakukan sebagai data, bukan instruksi. Pertanyaan di luar topik tomat, proses penyortiran, dan data
dashboard ditolak sebelum dikirim ke Gemini. Prompt membatasi jawaban pada data yang tersedia dan melarang
mengarang hasil; namun keluaran model tetap perlu diverifikasi. Dibatasi 20 pertanyaan/menit dan 1000 karakter/pesan.
Model diatur di `dashboard.chat.model`.

## Keamanan

- Belum ada login. Jalankan di `127.0.0.1` (bawaan). Bila `host: 0.0.0.0` agar HP bisa membuka dashboard, siapa pun
  di jaringan yang sama bisa melihat dashboard dan memakai chatbot (memakai kuota API Anda). Ingest tetap tertutup
  untuk perangkat lain karena butuh loopback atau `DASHBOARD_INGEST_KEY`.
- Foto hanya dilayani dari folder `data/`. CSV mengamankan sel yang diawali `=`, `+`, `-`, `@`.

## Catatan desain

- Grade selalu tampil sebagai huruf, tidak hanya warna. Merah sengaja tidak dipakai untuk grade jelek karena grade C
  juga mencakup tomat mentah.
- Tomat tanpa objek valid (mis. hanya tangan) tidak dibalas controller dan hanya tercatat sebagai log WARN
  (`docs/ALUR_ALAT.md`); error firmware (mis. `GRADE_TIMEOUT`) ada di tabel `events`, bukan `tomatoes`.

## Belum dikerjakan

- Tombol Start/Stop/Reset (perlu jalur perintah dashboard -> controller; chatbot sengaja tidak punya jalur ini).
- Tab kalibrasi (foto latar, `mm_per_pixel`, ambang warna).
- Login, bila dashboard akan dibuka ke jaringan.
