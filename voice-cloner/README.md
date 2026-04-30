# Voice Cloner Backend (FastAPI)

FastAPI backend for cloning voices using OpenVoice, secured with JWT bearer auth and backed by DynamoDB for user verification. It exposes REST and WebSocket endpoints for managing per-user "sessions", registering a reference voice, and generating cloned audio.

---

## Architecture Overview

- **Framework**: FastAPI
- **Auth & User Store**:
  - JWT bearer tokens (signed with a shared `JWT_SECRET`, typically issued by your auth provider).
  - Each request decodes the JWT, validates:
    - `exp` (token not expired).
    - `sub` and `email` claims are present.
  - Dynamodb table is queried by `sub`; the row's `email` must match the token's `email`.
- **Modules**
  - `main.py`
    - Creates the FastAPI `app`.
    - Registers CORS middleware.
    - Defines all REST and WebSocket routes:
      - `GET /health`
      - `GET /state`
      - `POST /train`
      - `POST /clone`
      - `POST /clone-sub` (no auth)
      - `WS /train`
      - `WS /clone`
      - `WS /clone-sub` (no auth)
    - Maintains an **in-memory per-user session store** keyed by `sub`, with:
      - `trained_voice_id`
      - `reference_b64` – aggregated base64 PCM reference audio (up to 20s)
      - `reference_path` – temp `.wav` file for the reference audio
      - `reference_target_se` – precomputed target speaker embedding from OpenVoice
      - `reference_audio_name` – audio name used when generating cloned audio
    - Maintains **sub_voice_links**: a map `sub` → `{ trained_voice_id, reference_target_se, reference_audio_name, expires_at }` used by `/clone-sub`. The link is created on the **first** `GET /state` call for a user who has a trained voice; it expires **1 hour** after that first `/state`. Subsequent `/state` calls do not extend the expiry.
    - **Embedding persistence** (optional): When `AWS_REGION` and `EMBEDDINGS_TABLE_NAME` are set, at the end of training (REST or WebSocket `operation: "end"`) the speaker embedding is stored in DynamoDB (keyed by `user_id` = `sub`), with `embedding_b64`/`embedding_shape` and optionally `reference_audio_s3_key`. For cloning and for authenticated `/state`, if the in-memory session has no embedding, the server loads it from DynamoDB into the session before proceeding.
    - **Reference audio in S3** (optional): When `S3_BUCKET` is set, at the end of training the reference speaker WAV is uploaded to S3 and its key is stored in DynamoDB and session. On clone (REST, WebSocket, or `/clone-sub`), the server downloads that reference to a temp file and passes it as `reference_speaker` to the cloner (reference speaker cannot be empty). If S3 is not configured or the reference key is missing, clone returns 422.
  - `auth.py`
    - `verify_jwt_and_load_user(token)`:
      - Decodes JWT using `JWT_SECRET` and `JWT_ALGORITHM`.
      - Ensures token has not expired and has `sub` and `email`.
      - Checks DynamoDB table (partition key `sub`) has an item whose `email` matches.
      - Returns an `AuthedUser` Pydantic model on success or raises an `HTTPException`.
    - `get_current_user`:
      - FastAPI dependency for REST endpoints using `HTTPBearer` auth.
    - `extract_token_from_websocket`:
      - Extracts a bearer token from `Authorization: Bearer <token>` header or `?token=<JWT>` query param.
  - `models.py`
    - Pydantic models for requests/responses:
      - `TrainRequest`, `TrainResponse`
      - `CloneRequest`, `CloneResponse`, `CloneSubRequest`
      - `StateResponse`, `HealthResponse`
      - `AuthedUser`
  - `audio.py`
    - `compute_trained_voice_id(reference_b64)`: `sha256` hash of the base64 audio, first 6 hex chars.
    - `append_pcm_chunk(existing_b64, new_chunk_b64)`: concatenates base64 PCM chunks per-session, enforcing a maximum duration (default 20 seconds).
    - `write_b64_audio_to_temp_wav(prefix, audio_b64)`: decodes base64 **PCM16** audio and writes a temp `.wav` file with the configured sample rate/channels/width, truncating to the max duration.
  - `cloner.py`
    - Thin wrapper around OpenVoice to perform the actual cloning:
      - `extract_speaker_embedding(audio_path) -> (se, audio_name)` for a given reference audio.
      - `clone(reference_speaker, base_speaker, target_se=None, target_audio_name=None) -> base64_pcm`
        - Generates a WAV via OpenVoice, reads the PCM frames, base64-encodes them, deletes the file, and returns **base64-encoded PCM**.
      - Used by `/clone` REST and WebSocket endpoints.

> **Note**: Session state is in-memory and keyed by `sub`; for multi-instance or long-lived production deployments you will likely want to move this to a shared store (Redis, DynamoDB, etc.).

---

## Environment Variables

Set the following environment variables for both local and production:

- **JWT & Auth**
  - `JWT_SECRET` – symmetric key for verifying JWT signatures.
  - `JWT_ALGORITHM` – algorithm (default `HS256`).
- **DynamoDB**
  - `DDB_TABLE_NAME` – table for user verification (auth):
    - Partition key `user_id` (string; stores the JWT `sub`).
    - Attribute `email` (string).
  - `AWS_REGION` – AWS region for DynamoDB tables (e.g. `us-east-1`).
  - `EMBEDDINGS_TABLE_NAME` – (optional) table for storing speaker embeddings; default `brivva-users-embeddings`. When set (with `AWS_REGION`), embeddings are persisted at end of training and loaded when cloning or when fetching state if the in-memory session has no embedding. Schema:
    - Partition key `user_id` (string; same as user `sub`).
    - Attributes: `trained_voice_id` (string), `embedding_b64` (string), `embedding_shape` (list of ints), `reference_audio_s3_key` (string; S3 key of the reference WAV when S3 is configured).
  - Standard AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, etc.) must be available to the process.
- **S3**
  - `S3_BUCKET` – bucket where reference speaker WAVs are stored. When set (with `AWS_REGION`), at the end of training the reference audio is uploaded to `s3://{S3_BUCKET}/{S3_REF_PREFIX}/{sub}/{trained_voice_id}.wav` and the key is stored in the embeddings table and session. On clone, the server downloads the reference to a temp file and passes it to the cloner (reference speaker cannot be empty).
  - `S3_REF_PREFIX` – optional key prefix for reference audio (default `voice-refs`).
- **Server**
  - `PORT` – optional, port FastAPI/uvicorn will bind to (defaults to `8000`).
  - `LOG_LEVEL` – optional, logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default `INFO`).
  - `LOG_FORMAT` – optional, format string for log lines (default includes timestamp, level, logger name, message).
  - `TORIO_USE_FFMPEG=0` – set in `main.py` before torch loads so torchaudio doesn’t use its FFmpeg extension. That extension is built for FFmpeg 6 (`libavutil.58`); if you have FFmpeg 7/8 (or no FFmpeg), it fails to load. With this disabled, torchaudio uses other backends; OpenVoice uses librosa/pydub for audio I/O.
  - `PCM_SAMPLE_RATE` – (optional) PCM sample rate in Hz (default `16000`).
  - `PCM_CHANNELS` – (optional) number of PCM channels (default `1`).
  - `PCM_SAMPLE_WIDTH` – (optional) bytes per PCM sample (default `2`, i.e. 16-bit).
  - `MAX_AUDIO_SECONDS` – (optional) maximum duration for reference/base audio (default `20`).

---

## Local Development Setup

### 1. Install Python dependencies

Create a virtual environment (recommended) and install dependencies:

```bash
cd voice-cloner
python -m venv .venv
source .venv/bin/activate  # on Windows use: .venv\Scripts\activate

pip install fastapi uvicorn[standard] boto3 PyJWT pydantic
# plus whatever OpenVoice requires, for example:
pip install torch  # and other openvoice deps as needed
```

If you prefer, add these to a `requirements.txt` and install via `pip install -r requirements.txt`.

### 2. Configure environment variables

Create a `.env` file or export vars in your shell, for example:

```bash
export JWT_SECRET="your-dev-secret"
export JWT_ALGORITHM="HS256"
export DDB_TABLE_NAME="voice-cloner-users-dev"
export AWS_REGION="us-east-1"
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
```

Ensure your DynamoDB table for auth exists and has at least:

- Partition key `user_id` (string; holds the JWT `sub`).
- Attribute `email` (string).

Optional: create the embeddings table (e.g. `brivva-users-embeddings`) with partition key `user_id` (string) so the server can persist and load speaker embeddings across restarts.

### 3. Download the latest checkpoint for inferencing

From OpenVoice: [https://github.com/myshell-ai/OpenVoice/blob/main/docs/USAGE.md](https://github.com/myshell-ai/OpenVoice/blob/main/docs/USAGE.md)  
Download the OpenVoice V2 checkpoint and save it as checkpoints_v2

### 4. Run the app locally

Start with uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The OpenAPI docs will be available at `http://localhost:8000/docs`.

---

## Production Deployment

Below is a generic production approach; adapt to your platform (ECS, EKS, Lambda, Cloud Run, etc.).

### 1. Containerization (recommended)

Create a `Dockerfile` such as:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir fastapi uvicorn[standard] boto3 PyJWT pydantic torch  # plus openvoice deps

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and push:

```bash
docker build -t your-registry/voice-cloner:latest .
docker push your-registry/voice-cloner:latest
```

### 2. Runtime configuration

At deployment time, set environment variables:

- `JWT_SECRET`, `JWT_ALGORITHM`
- `DDB_TABLE_NAME`, `AWS_REGION`
- `EMBEDDINGS_TABLE_NAME` (optional; default `brivva-users-embeddings`) if you want embedding persistence.
- AWS credentials via:
  - IAM Role for the task/pod/instance (preferred), or
  - Environment variables or secrets manager.

### 3. Scaling & session storage

- This implementation uses **in-memory sessions**, keyed by `sub`. In a multi-instance or auto-scaled environment:
  - All requests for a user must be routed to the same instance (sticky sessions), **or**
  - Replace the in-memory `sessions` dict with a shared store (e.g. Redis or DynamoDB).
- When **embedding persistence** is enabled (`EMBEDDINGS_TABLE_NAME` and `AWS_REGION` set), speaker embeddings are stored in DynamoDB at the end of training and loaded from DynamoDB when cloning or when fetching state if the current process has no embedding in session. That allows clone and state to work even when the user hits a different instance or after a restart, without requiring a shared session store.

### 4. Reverse proxy

Run behind a reverse proxy (e.g. Nginx, ALB) that:

- For REST:
  - Proxies `Authorization: Bearer <token>` header.
- For WebSockets:
  - Supports WebSocket upgrades and passes through `Authorization` header or a `token` query parameter.

---

## API Usage

All endpoints except `/health` require a valid JWT bearer token:

- REST: `Authorization: Bearer <JWT>`
- WebSockets:
  - `Authorization: Bearer <JWT>` header, **or**
  - `?token=<JWT>` query param.

The JWT must:

- Be signed with `JWT_SECRET` using `JWT_ALGORITHM`.
- Include:
  - `sub` – user identifier.
  - `email` – user email.
  - `exp` – expiration timestamp.

### 1. Health Check

- **Endpoint**: `GET /health`
- **Auth**: none
- **Response**:

```json
{
  "status": "ok"
}
```

### 2. Session State

- **Endpoint**: `GET /state`
- **Auth**: bearer token required
- **Response**:

```json
{
  "session_id": "user-sub",
  "trained_voice_id": "abc123" // or null
}
```

On the **first** `/state` call where the user has a trained voice, the server registers that `sub` for **clone-by-sub** usage: the link expires **1 hour** after this first `/state`. Later `/state` calls do not extend the expiry. If the session has no embedding in memory, the server attempts to load it from the embeddings DynamoDB table (when configured) so the link can be created.

Example:

```bash
curl -H "Authorization: Bearer $JWT" http://localhost:8000/state
```

### 3. Train (register reference voice)

Training uses a three-phase flow: **start** → **reference** chunks → **end**. Send `{"operation": "start"}` to begin, one or more `{"reference": "<base64-pcm>"}` to append audio (up to 20s), then `{"operation": "end"}` to run training and persist the voice. When embedding persistence is configured, the speaker embedding is also stored in the `brivva-users-embeddings` DynamoDB table at **end** (for both REST and WebSocket).

#### REST

- **Endpoint**: `POST /train`
- **Auth**: bearer token required
- **Request bodies** (one of these per request):
  - **Start**: `{"operation": "start"}` — clears the session’s reference buffer and starts a new training round.
  - **Chunk**: `{"reference": "<base64-encoded-pcm-chunk>"}` — appends the chunk (only valid after `start` and before `end`). Buffer is capped at 20s.
  - **End**: `{"operation": "end"}` — runs training on the accumulated buffer (write WAV, extract embedding, store) and returns the trained voice id.
- **Responses**:
  - After **start**: `{"status": "started"}`
  - After **chunk**: `{"status": "chunk_received"}`
  - After **end**: `{"session_id": "user-sub", "trained_voice_id": "abc123"}`
- **Errors**: 422 if you send `reference` before `start`, or `end` with no chunks.

Example:

```bash
curl -X POST http://localhost:8000/train -H "Authorization: Bearer $JWT" -H "Content-Type: application/json" -d '{"operation": "start"}'
curl -X POST http://localhost:8000/train -H "Authorization: Bearer $JWT" -H "Content-Type: application/json" -d '{"reference": "'"$REFERENCE_B64"'"}'
curl -X POST http://localhost:8000/train -H "Authorization: Bearer $JWT" -H "Content-Type: application/json" -d '{"operation": "end"}'
```

#### WebSocket

- **Endpoint**: `WS /train`
- **Auth**: Header `Authorization: Bearer <JWT>` or query `?token=<JWT>`
- **Client → Server messages** (in order):
  1. `{"operation": "start"}` — begin chunking
  2. `{"reference": "<base64-pcm-chunk>"}` — append chunks (repeat as needed, up to 20s)
  3. `{"operation": "end"}` — finish and run training
- **Server → Client**:
  - After start: `{"type": "started"}`
  - After chunk: `{"type": "chunk_received"}`
  - After end: `{"type": "train_result", "session_id": "user-sub", "trained_voice_id": "abc123"}`
- **Errors** (WebSocket JSON): e.g. `{"error": "Send {'operation': 'start'} first.", "code": 422}`, `{"error": "Invalid JSON", "code": 400}`

### 4. Clone (generate cloned voice)

Cloning requires an existing reference voice registered via `/train`.

#### REST

- **Endpoint**: `POST /clone`
- **Auth**: bearer token required
- **Behavior**:
  - Accepts **base64-encoded PCM16** chunks in the `base` field.
  - Each request is treated independently (no accumulation); audio longer than `MAX_AUDIO_SECONDS` is truncated.
  - Uses the precomputed reference speaker embedding from the session; if the session has no embedding (e.g. new process or different instance), the server loads it from the embeddings DynamoDB table when configured.
- **Request Body**:

```json
{
  "base": "<base64-encoded-pcm-audio>"
}
```

- **Responses**:

Success:

```json
{
  "session_id": "user-sub",
  "trained_voice_id": "abc123",
  "output_path": "<base64-encoded-pcm-of-cloned-voice>"
}
```

Errors:

- `422 Unprocessable Entity` – if no trained voice is registered for this user:

```json
{
  "detail": "No trained voice registered for this session"
}
```

- `500 Internal Server Error` – if the cloning process fails.

Example:

```bash
curl -X POST http://localhost:8000/clone \
  -H "Authorization: Bearer $JWT" \
  -H "Content-Type: application/json" \
  -d '{"base": "'"$BASE_B64"'"}'
```

#### WebSocket

- **Endpoint**: `WS /clone`
- **Auth**:
  - Header: `Authorization: Bearer <JWT>`, or
  - Query: `/clone?token=<JWT>`
- **Client → Server message**:

```json
{
  "base": "<base64-encoded-pcm-audio>"
}
```

- **Server → Client success message**:

```json
{
  "type": "clone_result",
  "session_id": "user-sub",
  "trained_voice_id": "abc123",
  "output_path": "<base64-encoded-pcm-of-cloned-voice>"
}
```

- **Error responses**:
  - No trained voice: `{"error": "No trained voice registered for this session", "code": 422}`
  - Invalid JSON: `{"error": "Invalid JSON", "code": 400}`
  - Cloning failure: `{"error": "Failed to generate cloned voice: ...", "code": 500}`

### 5. Clone by sub (`/clone-sub`)

Clone using a **sub** to resolve the voice instead of the authenticated user. The sub must have been registered by a **first** `GET /state` call (see §2); the link expires **1 hour** after that first `/state`. No JWT is required.

**Typical flow**: User A trains a voice and calls `GET /state` → their `sub` is registered for 1 hour. Anyone (e.g. a different client or user) can call `POST /clone-sub` or `WS /clone-sub` with User A’s `sub` and the base audio until the link expires.

#### REST

- **Endpoint**: `POST /clone-sub`
- **Auth**: none
- **Request body**:

```json
{
  "sub": "user-sub-from-state",
  "base": "<base64-encoded-pcm-audio>"
}
```

- **Success response**: same as `POST /clone`:

```json
{
  "session_id": "user-sub-from-state",
  "trained_voice_id": "abc123",
  "output_path": "<base64-encoded-pcm-of-cloned-voice>"
}
```

- **Errors**:
  - `422` – no voice link for this sub or link expired: `{"detail": "No voice link for this sub or link expired"}` or `{"detail": "Voice link expired"}`
  - `500` – cloning failed (same as `/clone`)

Example:

```bash
curl -X POST http://localhost:8000/clone-sub \
  -H "Content-Type: application/json" \
  -d '{"sub": "user-sub-from-state", "base": "'"$BASE_B64"'"}'
```

#### WebSocket

- **Endpoint**: `WS /clone-sub`
- **Auth**: none
- **Client → Server message**:

```json
{
  "sub": "user-sub-from-state",
  "base": "<base64-encoded-pcm-audio>"
}
```

- **Server → Client success**: same shape as `WS /clone`:

```json
{
  "type": "clone_result",
  "session_id": "user-sub-from-state",
  "trained_voice_id": "abc123",
  "output_path": "<base64-encoded-pcm-of-cloned-voice>"
}
```

- **Error responses**: `{"error": "<message>", "code": 422|400|500}` (e.g. missing sub/base, invalid JSON, no link or expired, cloning failure).

---

## Testing Tips

- Use `http://localhost:8000/docs` to interactively test REST endpoints and view models.
- Use a WebSocket client (e.g. `wscat`, browser dev tools, or Postman) to test `/train` and `/clone` WebSocket flows.
- Make sure the JWTs you generate:
  - Are signed with the same `JWT_SECRET` and `JWT_ALGORITHM`.
  - Have `sub` and `email` claims matching a row in your DynamoDB table.

