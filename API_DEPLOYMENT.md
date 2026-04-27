# MuseTalk API, security, and worker ↔ GPU contract

This repository runs the **lip-sync GPU service** (Gradio UI + MuseTalk models + FastAPI). A separate **CPU mux worker** (not in this repo) muxes audio and video, then sends the muxed MP4 **to this server** for lip-sync.

The Gradio UI and HTTP API share one process (`app.py`). FastAPI (Uvicorn) serves REST routes; Gradio is mounted at `/`.

## Environment variables (this server)

| Variable | Required | Description |
|----------|----------|-------------|
| `SECURED_MODE` | No | `on` / `true` / `1` → Gradio requires HTTP Basic login. Any other value or unset → open Gradio UI. |
| `GRADIO_USER` | If `SECURED_MODE` on | Gradio login username (recommended; avoids clashing with the OS `USER` variable). |
| `GRADIO_PASS` | If `SECURED_MODE` on | Gradio login password. |
| `USER` | Fallback | Used as username only if `GRADIO_USER` is empty. On Unix, `USER` is often already set to your shell login—prefer `GRADIO_USER` in production. |
| `PASS` | Fallback | Used as password if `GRADIO_PASS` is empty. |
| `BEARER_TOKEN` | No | If set, `/api/job*`, `POST /job`, and `GET /job` require `Authorization: Bearer <token>`. If unset, those routes are open (suitable only for trusted networks). |
| `GPU_MULTIPART_FIELD` | No | Multipart **field name** that **clients** must use when `POST`ing the muxed file to **this** server (default `file`). |
| `API_JOB_DIR` | No | Directory for per-job uploads and metadata (default `./results/api_jobs`). |

## External CPU worker

The mux worker sets **`GPU_SERVER_URL`** to the **base URL of this MuseTalk deployment** (for example `https://musetalk.internal:7860`). It does **not** set env vars on this server for that URL—`GPU_SERVER_URL` is only meaningful **on the worker** that calls into here.

Worker flow (implemented outside this repo):

1. `POST {GPU_SERVER_URL}/job?id=<job_id>` with `multipart/form-data` and one file field (`GPU_MULTIPART_FIELD` on **this** server, default `file`) containing the muxed MP4.
2. Poll `GET {GPU_SERVER_URL}/job?id=<job_id>` every **2 seconds** for up to **5 minutes**.
3. Treat a final response as:
   - `200` with `Content-Type: application/octet-stream` or `video/*` → body is the final lip-synced video.
   - `200` with `application/json` → for in-flight jobs this server returns `{"status":"processing"}`; on failure `{"status":"error","message":"..."}`.

While the job is running, `GET /job?id=...` returns JSON `{"status":"processing"}`. When finished, the same URL returns the MP4 bytes (`application/octet-stream`).

## Modes for administration

### 1. Local / internal (default)

- Unset `SECURED_MODE` or set `SECURED_MODE=off`.
- Leave `BEARER_TOKEN` unset only if the network is trusted.
- Point external workers at this host via their own `GPU_SERVER_URL`.

### 2. Public UI, locked HTTP APIs

- `SECURED_MODE=off`
- `BEARER_TOKEN=<long random secret>`

Gradio stays open; workers and `/api/job` clients must send the bearer token on `POST /job`, `GET /job`, and `/api/job*`.

### 3. Locked UI and APIs

- `SECURED_MODE=on`
- `GRADIO_USER` / `GRADIO_PASS` set (recommended)
- `BEARER_TOKEN=<long random secret>`

Gradio uses HTTP Basic auth. Workers use the bearer token (not the Gradio password).

## HTTP API on this server

Base URL: same host/port as the app (e.g. `http://127.0.0.1:7860`).

### Worker contract (lip-sync GPU role)

- `POST /job?id=<job_id>` — bearer if `BEARER_TOKEN` is set. Multipart field name from **`GPU_MULTIPART_FIELD`** (default `file`). Body = muxed MP4. Response `202` JSON `{"status":"queued","id":"<job_id>"}`.
- `GET /job?id=<job_id>` — bearer if configured. Until done: `200` JSON `{"status":"processing"}`. On success: `200` binary MP4 (`application/octet-stream`). On failure: `200` JSON `{"status":"error","message":"..."}`.

`job_id` must match `^[a-zA-Z0-9._-]+$` (max 256 characters). After a successful job, reuse the same id only after the worker uses a **new** id (this server returns `409` if a completed id is posted again).

### Native REST (separate audio + video uploads)

- `GET /api/health` — no bearer; `{"status":"ok"}`.
- `POST /api/job` — bearer if set; multipart `audio` + `video` plus optional tuning form fields; `202` with `job_id`.
- `GET /api/job/{job_id}` — status JSON.
- `GET /api/job/{job_id}/download` — MP4 when `done`.

### Example: worker-style call (curl)

```bash
export BASE=http://127.0.0.1:7860
export TOKEN=your-bearer-secret
export JOB=my-job-001

curl -sS -X POST "$BASE/job?id=$JOB" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@muxed.mp4"

curl -sS -H "Authorization: Bearer $TOKEN" "$BASE/job?id=$JOB" -o out.mp4
```

Poll in a loop until the response is binary (or JSON `status` is `error`).

### Example: native `/api/job` (curl)

```bash
curl -sS -X POST "$BASE/api/job" \
  -H "Authorization: Bearer $TOKEN" \
  -F "audio=@speech.wav" \
  -F "video=@reference.mp4" \
  -F "bbox_shift=0" \
  -F "parsing_mode=jaw"
```

## Run command

```bash
python app.py --use_float16 --ip 0.0.0.0 --port 7860
```

Docker uses `--ip 0.0.0.0 --port 7860` by default so the service listens on all interfaces.

## Implementation notes

- `POST /job` stores the muxed file, **demuxes** it with ffmpeg into a reference video + driving audio, then runs the same MuseTalk `inference` path as the Gradio app (default bbox/parsing parameters: `bbox_shift=0`, `extra_margin=10`, `jaw`, cheek widths `90`).
- If you need different tuning for contract jobs, extend the handler or use `/api/job` with explicit `audio` + `video` files instead of muxed upload.
