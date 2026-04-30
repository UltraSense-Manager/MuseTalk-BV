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
| `BEARER_TOKEN` | No | Admin override bearer token. If request token equals this value, API access is allowed. |
| `JWT_SECRET` | Conditional | Required for non-admin bearer auth. Used to verify JWT signatures for `/api/job*`, `/api/realtime/job`, `POST /job`, and `GET /job`. |
| `JWT_ALGORITHM` | No | JWT algorithm (default `HS256`). |
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

- `GET /api/health` — no bearer; `{"status":"ok", ...}`. When the voice cloner is mounted, adds `voice_cloner`, `voice_cloner_prefix`, `voice_cloner_auth` (`ddb` \| `jwt`), and `voice_cloner_backend` (`aws` \| `local`, inferred from env).
- `POST /api/job` — bearer if set; multipart `audio` + `video` plus optional tuning form fields; `202` with `job_id`.
  - **`resolution_scale`** (string, default `full`) — `full` (100%), `half` (50%), `eighth` (12.5%), `lowest` (~6.25% linear). Frames are processed at the reduced size for speed, then **upscaled to the original video resolution** before the final MP4 is written for download.
- `GET /api/job/{job_id}` — status JSON.
- `GET /api/job/{job_id}/download` — MP4 when `done`.

### Voice clone (OpenVoice, monolith)

When **`ENABLE_VOICE_CLONER=on`** and **`JWT_SECRET`** is set, the in-repo **voice-cloner** FastAPI app is mounted at **`/api/voice`** on the same process and port as MuseTalk (same bearer/JWT as other API routes).

- **Train**: `POST /api/voice/train` — same JSON contract as standalone voice-cloner (`operation: start|end`, `reference` base64 PCM chunks). Returns `trained_voice_id` on `end`.
- **Clone**: `POST /api/voice/clone` — body `{"base":"<base64 PCM16 mono 16kHz>"}`; returns `output_path` (base64 PCM of cloned audio).
- **State**: `GET /api/voice/state` — same as upstream (optional bearer or `?sub=` link flow).

**Persistence without AWS**: If DynamoDB user table env (`DDB_TABLE_NAME` + `AWS_REGION`) is not configured, the cloner verifies **JWT only** (claims `sub` required; `email` optional, defaults for local). If the embeddings DynamoDB table is unavailable, embeddings are kept **in memory** (single-process). If **S3** is not configured, reference WAVs are stored under **`VOICE_CLONER_LOCAL_DIR`** (default `./results/voice_cloner_data`) so cloning still works locally.

With full AWS, behavior matches production-style DynamoDB + optional S3 for reference audio.

### Realtime pipeline (`/api/realtime/job`) — deprecated for new clients

**Deprecation:** New integrations should prefer **`POST /api/job`** (standard pipeline). The bundled **lipsync-rt-demo** no longer calls `/api/realtime/job`. The endpoint remains available for backward compatibility.

Same job lifecycle as `/api/job` (`GET /api/job/{job_id}`, `/download`). Uses MuseTalk’s **realtime** path: the first **`realtime_prep_frames`** frames (default **30**) are extracted from the uploaded video with **ffmpeg**, then avatar preparation (landmarks, latents, masks) and batched inference (as in `scripts/realtime_inference.py`), then ffmpeg muxes frames + driving audio.

Prepared materials are stored under **`{API_JOB_DIR}/realtime_avatars/{user_id}/`** so you can run again **without** re-uploading video or re-running prep.

- `POST /api/realtime/job` — multipart **`audio`** (required). **`video`** required for a **new** prep run; omit **`video`** when reusing (see `use_clone=true`).
  - Same tuning as `/api/job`: `bbox_shift`, `extra_margin`, `parsing_mode`, `left_cheek_width`, `right_cheek_width`
  - `realtime_prep_frames` (int, default `30`, clamped 1–300) — used only when `use_clone=false`
  - `realtime_batch_size` (int, default `20`, clamped 1–128)
  - `realtime_fps` (int, default `25`, clamped 1–60)
  - **`use_clone`** (form boolean, default `false`) — if `true`, skips video prep and loads latents/masks from persisted clone materials.
  - **`clone_id`** (form string, optional) — clone identifier to use. If omitted/null, server resolves to decoded JWT `sub`/`uid`. Must match `^[a-zA-Z0-9._-]+$`.
  - **`resolution_scale`** — same presets as `/api/job`. For **new** prep, prep frames are downscaled before landmark/latent work; the final muxed MP4 is upscaled to the **original extracted frame size** before download. For **`use_clone=true`**, upscale uses the width/height stored when that avatar was first prepared (ignored for prep if already on disk).
- Response **`202`** JSON includes `job_id`, `user_id`, `clone_id`, `"kind": "realtime"`, `realtime_prep_frames`, and `use_clone`.
- **`GET /api/job/{job_id}`** includes `user_id` and `clone_id` for realtime jobs when known.

### Auth behavior (API routes)

- Requests must send `Authorization: Bearer <token>`.
- If token exactly equals `BEARER_TOKEN`, request is accepted via admin override.
- Otherwise, token is validated as JWT using `JWT_SECRET` + `JWT_ALGORITHM`, requiring `exp`.
- Realtime clone default identity comes from JWT (`sub`, fallback `uid`) when `clone_id` is not provided.

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
