# Voice Cloner Test Frontend

Minimal web UI to exercise the Voice Cloner API: state (REST), train from mic (WebSocket), and TTS → clone (WebSocket).

## Run locally

1. Serve this folder over HTTP (required for mic and optional tab capture).

   ```bash
   cd test-frontend
   python -m http.server 8080
   # or: npx serve -p 8080
   ```

2. Open `http://localhost:8080` in your browser.

3. Ensure the Voice Cloner backend is running (e.g. `uvicorn main:app` in the parent project) and CORS is enabled.

## Usage

- **Config**: Set **API base URL** (e.g. `http://localhost:8000`) and paste your **JWT**.
- **State**: Click **Get state** to call `GET /state` and show `session_id` and `trained_voice_id`.
- **Train**: Click **Start training (mic)** to stream mic audio as base64 PCM over `WS /train`; stop when you have enough (up to 20s).
- **Clone**: Enter **TTS text**, then **TTS & clone**. The app will ask to share **this tab** with audio, speak the text with the browser TTS, record it, send it as base audio to `WS /clone`, then play the returned base64 PCM.

## Audio format

- Mic and TTS capture are converted to **16 kHz mono 16‑bit PCM** before sending, matching the backend’s default `PCM_*` settings.
- Clone response is base64 PCM at the same format; the app builds a WAV and plays it in the page.
