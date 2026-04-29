import { useCallback, useEffect, useRef, useState } from "react";
import {
  downloadJob,
  pollUntilDone,
  submitRealtimeJob,
  submitStandardJob,
} from "./api";
import {
  formatTtsDuration,
  getDecodedAudioDurationSeconds,
  previewTextToSpeech,
  puterAvailable,
  textToSpeechFile,
  validateTtsText,
  waitAudioElementDuration,
} from "./tts";
import "./App.css";

const POLL_INTERVAL_MS = 2000;
const JOB_DEADLINE_MS = 3600 * 1000;

function videoFileKey(f: File | null): string | null {
  if (!f) return null;
  return `${f.name}:${f.size}:${f.lastModified}`;
}

export default function App() {
  const [baseUrl, setBaseUrl] = useState("");
  const [bearerToken, setBearerToken] = useState("");
  const [ttsText, setTtsText] = useState("");
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoObjectUrl, setVideoObjectUrl] = useState<string | null>(null);

  const [realtime, setRealtime] = useState(true);
  const [prepFrames, setPrepFrames] = useState(30);
  /** full | half | eighth | lowest — server downscales processing then upscales for download */
  const [resolutionScale, setResolutionScale] = useState("full");

  const [busy, setBusy] = useState(false);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewActive, setPreviewActive] = useState(false);
  const [ttsDurationLine, setTtsDurationLine] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [liveElapsedMs, setLiveElapsedMs] = useState(0);

  const lastCloneIdRef = useRef<string | null>(null);
  const lastRealtimeVideoKeyRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const runStartRef = useRef<number>(0);
  const previewAudioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    if (!videoFile) {
      setVideoObjectUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return null;
      });
      return;
    }
    const url = URL.createObjectURL(videoFile);
    setVideoObjectUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return url;
    });
    return () => URL.revokeObjectURL(url);
  }, [videoFile]);

  useEffect(() => {
    setTtsDurationLine(null);
  }, [ttsText]);

  useEffect(() => {
    return () => {
      const a = previewAudioRef.current;
      if (a) {
        a.pause();
        previewAudioRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!busy) return;
    const id = window.setInterval(() => {
      setLiveElapsedMs(performance.now() - runStartRef.current);
    }, 100);
    return () => clearInterval(id);
  }, [busy]);

  const displayElapsed =
    busy && runStartRef.current ? liveElapsedMs / 1000 : elapsedMs !== null ? elapsedMs / 1000 : null;

  const currentVideoKey = videoFileKey(videoFile);
  const willReuseClone =
    realtime &&
    !!lastCloneIdRef.current &&
    !!currentVideoKey &&
    lastRealtimeVideoKeyRef.current === currentVideoKey;

  const validate = useCallback((): string | null => {
    const base = baseUrl.trim();
    if (!base) return "Enter the API base URL (e.g. http://127.0.0.1:7860).";
    const normalized = base.startsWith("http") ? base : `http://${base}`;
    try {
      void new URL(normalized);
    } catch {
      return "API base URL is not valid.";
    }
    const ttsErr = validateTtsText(ttsText);
    if (ttsErr) return ttsErr;
    if (!puterAvailable()) return "Puter.js is still loading or blocked. Refresh the page.";
    if (!realtime) {
      if (!videoFile) return "Choose a reference video for standard mode.";
      return null;
    }
    if (!willReuseClone && !videoFile) {
      return "Choose a reference video (or reuse the same file as the last successful realtime run).";
    }
    return null;
  }, [baseUrl, ttsText, videoFile, realtime, willReuseClone]);

  const stopPreview = useCallback(() => {
    const a = previewAudioRef.current;
    if (a) {
      a.pause();
      a.currentTime = 0;
      previewAudioRef.current = null;
    }
    setPreviewActive(false);
  }, []);

  const onPreviewTts = async () => {
    setError(null);
    setInfo(null);
    const ttsErr = validateTtsText(ttsText);
    if (ttsErr) {
      setError(ttsErr);
      return;
    }
    if (!puterAvailable()) {
      setError("Puter.js is still loading or blocked. Refresh the page.");
      return;
    }
    setPreviewBusy(true);
    try {
      stopPreview();
      const el = await previewTextToSpeech(ttsText);
      previewAudioRef.current = el;
      el.onended = () => {
        if (previewAudioRef.current === el) previewAudioRef.current = null;
        setPreviewActive(false);
      };
      let durSec: number | null = null;
      try {
        durSec = await waitAudioElementDuration(el);
        setTtsDurationLine(`TTS duration: ${formatTtsDuration(durSec)} (preview)`);
      } catch {
        setTtsDurationLine(null);
      }
      await el.play();
      setPreviewActive(true);
      setInfo(
        durSec !== null
          ? `Playing TTS preview (${formatTtsDuration(durSec)})…`
          : "Playing TTS preview…"
      );
    } catch (err) {
      previewAudioRef.current?.pause();
      previewAudioRef.current = null;
      setPreviewActive(false);
      const msg = err instanceof Error ? err.message : String(err);
      setError(`Preview failed: ${msg}`);
      setInfo(null);
    } finally {
      setPreviewBusy(false);
    }
  };

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setInfo(null);
    const v = validate();
    if (v) {
      setError(v);
      return;
    }

    const base = baseUrl.trim().startsWith("http")
      ? baseUrl.trim()
      : `http://${baseUrl.trim()}`;

    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;

    stopPreview();

    setBusy(true);
    runStartRef.current = performance.now();
    setElapsedMs(null);
    setLiveElapsedMs(0);

    try {
      setInfo("Synthesizing speech (Puter.js)…");
      const audioFile = await textToSpeechFile(ttsText);
      try {
        const dur = await getDecodedAudioDurationSeconds(audioFile);
        setTtsDurationLine(`TTS duration: ${formatTtsDuration(dur)} (this run)`);
        setInfo(`Driving audio ${formatTtsDuration(dur)} — submitting job…`);
      } catch {
        setTtsDurationLine(null);
        setInfo("Submitting job…");
      }

      const commonForm: Record<string, string> = {
        bbox_shift: "0",
        extra_margin: "10",
        parsing_mode: "jaw",
        left_cheek_width: "90",
        right_cheek_width: "90",
        resolution_scale: resolutionScale,
      };

      setInfo("Submitting job…");

      if (!realtime) {
        if (!videoFile) throw new Error("Missing video.");
        const submit = await submitStandardJob(
          base,
          bearerToken,
          audioFile,
          videoFile,
          commonForm,
          ac.signal
        );
        setInfo(`Queued job ${submit.job_id}. Waiting…`);
        await pollUntilDone(base, bearerToken, submit.job_id, {
          intervalMs: POLL_INTERVAL_MS,
          deadlineMs: JOB_DEADLINE_MS,
          signal: ac.signal,
          onTick: (b) => setInfo(`Status: ${b.status}…`),
        });
        setInfo("Downloading result…");
        const blob = await downloadJob(base, bearerToken, submit.job_id, ac.signal);
        lastCloneIdRef.current = null;
        lastRealtimeVideoKeyRef.current = null;
        triggerBlobDownload(blob, "lipsync-output.mp4");
      } else {
        const prep = Math.min(200, Math.max(15, prepFrames));
        const rtForm: Record<string, string> = {
          ...commonForm,
          realtime_prep_frames: String(prep),
          realtime_batch_size: "20",
          realtime_fps: "25",
        };
        const reuseId = willReuseClone ? lastCloneIdRef.current! : "";
        if (reuseId) {
          rtForm.use_clone = "true";
          rtForm.clone_id = reuseId;
        } else {
          rtForm.use_clone = "false";
        }

        const videoForUpload = reuseId ? null : videoFile;

        const submit = await submitRealtimeJob(
          base,
          bearerToken,
          audioFile,
          rtForm,
          videoForUpload,
          ac.signal
        );

        const cid = submit.clone_id || submit.user_id;
        if (cid) setInfo(`Queued ${submit.job_id} (clone_id=${cid}). Waiting…`);
        else setInfo(`Queued ${submit.job_id}. Waiting…`);

        const doneBody = await pollUntilDone(base, bearerToken, submit.job_id, {
          intervalMs: POLL_INTERVAL_MS,
          deadlineMs: JOB_DEADLINE_MS,
          signal: ac.signal,
          onTick: (b) => {
            const av = b.clone_id || b.user_id;
            const avs = av ? ` clone_id=${av}` : "";
            setInfo(`Status: ${b.status}${avs}…`);
          },
        });

        const finalClone = doneBody.clone_id || doneBody.user_id || submit.clone_id || submit.user_id;
        setInfo("Downloading result…");
        const blob = await downloadJob(base, bearerToken, submit.job_id, ac.signal);

        if (finalClone) lastCloneIdRef.current = finalClone;
        if (!reuseId && videoFile) {
          lastRealtimeVideoKeyRef.current = videoFileKey(videoFile);
        }
        triggerBlobDownload(blob, "lipsync-realtime-output.mp4");
      }

      const elapsed = performance.now() - runStartRef.current;
      setElapsedMs(elapsed);
      setInfo(`Done in ${(elapsed / 1000).toFixed(1)}s.`);
    } catch (err) {
      const msg =
        err instanceof DOMException && err.name === "AbortError"
          ? "Cancelled."
          : err instanceof Error
            ? err.message
            : String(err);
      setError(msg);
      setInfo(null);
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  };

  return (
    <div className="app">
      <h1>Lipsync demo</h1>
      <p className="hint">
        Driving audio comes from Puter.js TTS. Configure your MuseTalk API URL and bearer token
        (same as <code>test.py</code>).
      </p>

      <form className="stack" onSubmit={onSubmit}>
        <div className="field">
          <label htmlFor="base">API base URL</label>
          <input
            id="base"
            type="url"
            autoComplete="off"
            placeholder="http://127.0.0.1:7860"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            disabled={busy}
          />
        </div>

        <div className="field">
          <label htmlFor="token">Bearer token (optional if server has none)</label>
          <input
            id="token"
            type="password"
            autoComplete="off"
            placeholder="Paste token"
            value={bearerToken}
            onChange={(e) => setBearerToken(e.target.value)}
            disabled={busy}
          />
        </div>

        <div className="field">
          <label htmlFor="video">Reference video</label>
          <input
            id="video"
            type="file"
            accept="video/*"
            disabled={busy}
            onChange={(e) => setVideoFile(e.target.files?.[0] ?? null)}
          />
          {videoObjectUrl ? (
            <video className="preview" src={videoObjectUrl} controls muted playsInline />
          ) : null}
        </div>

        <div className="field">
          <label htmlFor="tts">Text for TTS (Puter.js)</label>
          <textarea
            id="tts"
            value={ttsText}
            onChange={(e) => setTtsText(e.target.value)}
            disabled={busy}
            placeholder="Script to synthesize as driving audio…"
          />
          <div className="row tts-actions">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => void onPreviewTts()}
              disabled={busy || previewBusy}
            >
              {previewBusy ? "Preview…" : "Preview TTS"}
            </button>
            <button
              type="button"
              className="btn btn-ghost"
              onClick={stopPreview}
              disabled={busy || !previewActive}
              title="Stop preview playback"
            >
              Stop preview
            </button>
          </div>
          <p className="hint">Preview uses the same Puter synthesis as submit (WAV).</p>
        </div>

        <div className="row">
          <input
            id="rt"
            type="checkbox"
            checked={realtime}
            onChange={(e) => setRealtime(e.target.checked)}
            disabled={busy}
          />
          <label htmlFor="rt">Realtime mode (POST /api/realtime/job)</label>
        </div>

        <div className="field">
          <label htmlFor="res">Resolution scale (faster at lower; output upscaled to full)</label>
          <select
            id="res"
            value={resolutionScale}
            onChange={(e) => setResolutionScale(e.target.value)}
            disabled={busy}
          >
            <option value="full">100% full</option>
            <option value="half">50% (half)</option>
            <option value="eighth">12.5% (eighth)</option>
            <option value="lowest">Lowest (~6.25%)</option>
          </select>
          <p className="hint">
            Applies to both standard and realtime. Clone reuse uses the upscale target saved when the
            avatar was first prepared.
          </p>
        </div>

        <div className={`field ${realtime ? "" : "dimmed"}`}>
          <label htmlFor="prep">Realtime prep frames (first N frames)</label>
          <div className="slider-row">
            <input
              id="prep"
              type="range"
              min={15}
              max={200}
              value={prepFrames}
              onChange={(e) => setPrepFrames(Number(e.target.value))}
              disabled={busy || !realtime}
            />
            <span className="value">{prepFrames}</span>
          </div>
          <p className="hint">
            Backend clamps to 1–300. With realtime on, the same video file as your last successful
            run sends <code>clone_id</code> + <code>use_clone=true</code> and skips re-uploading video.
          </p>
        </div>

        {realtime && willReuseClone ? (
          <p className="msg info">
            Same video as last successful realtime job — request will reuse{" "}
            <code>clone_id</code> and omit the video part.
          </p>
        ) : null}

        <div className="row">
          <button className="btn" type="submit" disabled={busy}>
            {busy ? "Working…" : "Run lipsync"}
          </button>
          {busy ? (
            <button
              type="button"
              className="btn"
              style={{ background: "#444" }}
              onClick={() => abortRef.current?.abort()}
            >
              Cancel
            </button>
          ) : null}
        </div>

        {displayElapsed !== null ? (
          <p className="stopwatch">Elapsed: {displayElapsed.toFixed(1)}s</p>
        ) : null}

        {ttsDurationLine ? <p className="hint">{ttsDurationLine}</p> : null}

        {error ? (
          <div className="msg error" role="alert">
            {error}
          </div>
        ) : null}
        {info && !error ? <div className="msg info">{info}</div> : null}
      </form>
    </div>
  );
}

function triggerBlobDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
