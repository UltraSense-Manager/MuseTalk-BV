import { useCallback, useEffect, useRef, useState } from "react";
import { downloadJob, pollUntilDone, submitStandardJob } from "./api";
import {
  formatTtsDuration,
  getDecodedAudioDurationSeconds,
  previewTextToSpeech,
  puterAvailable,
  textToSpeechFile,
  validateTtsText,
  waitAudioElementDuration,
} from "./tts";
import {
  audioToMonoPcm16Base64,
  cloneVoiceRest,
  pcm16Base64ToWavBlob,
  trainVoiceRest,
} from "./voiceCloner";
import "./App.css";

const POLL_INTERVAL_MS = 2000;
const JOB_DEADLINE_MS = 3600 * 1000;

export default function App() {
  const [baseUrl, setBaseUrl] = useState("");
  const [bearerToken, setBearerToken] = useState("");
  const [ttsText, setTtsText] = useState("");
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoObjectUrl, setVideoObjectUrl] = useState<string | null>(null);

  const [resolutionScale, setResolutionScale] = useState("full");

  const [useVoiceClone, setUseVoiceClone] = useState(false);
  const [voiceRefFile, setVoiceRefFile] = useState<File | null>(null);
  const [trainedVoiceId, setTrainedVoiceId] = useState("");
  const [voiceBusy, setVoiceBusy] = useState(false);

  const [busy, setBusy] = useState(false);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewActive, setPreviewActive] = useState(false);
  const [ttsDurationLine, setTtsDurationLine] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [liveElapsedMs, setLiveElapsedMs] = useState(0);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordChunksRef = useRef<Blob[]>([]);
  const [recording, setRecording] = useState(false);

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
    if (!videoFile) return "Choose a reference video.";
    if (useVoiceClone) {
      if (!bearerToken.trim()) return "Voice clone requires a bearer token (same JWT as the API).";
      if (!trainedVoiceId.trim()) return "Register a reference voice first (train), or turn off voice clone.";
    }
    return null;
  }, [baseUrl, ttsText, videoFile, useVoiceClone, bearerToken, trainedVoiceId]);

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

  const normalizedBase = () =>
    baseUrl.trim().startsWith("http") ? baseUrl.trim() : `http://${baseUrl.trim()}`;

  const onRegisterVoice = async () => {
    setError(null);
    setInfo(null);
    const base = normalizedBase();
    if (!bearerToken.trim()) {
      setError("Bearer token required to train voice.");
      return;
    }
    let blob: Blob | null = null;
    if (voiceRefFile) {
      blob = voiceRefFile;
    } else {
      setError("Choose a reference audio file (WAV/MP3) or record, then train.");
      return;
    }
    setVoiceBusy(true);
    try {
      setInfo("Encoding reference audio…");
      const pcmB64 = await audioToMonoPcm16Base64(blob);
      setInfo("Training voice on server…");
      const vid = await trainVoiceRest(base, bearerToken, pcmB64, abortRef.current?.signal);
      setTrainedVoiceId(vid);
      setInfo(`Voice registered. trained_voice_id=${vid}`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setInfo(null);
    } finally {
      setVoiceBusy(false);
    }
  };

  const startRecordRef = () => {
    setError(null);
    void navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((stream) => {
        recordChunksRef.current = [];
        const mr = new MediaRecorder(stream);
        mediaRecorderRef.current = mr;
        mr.ondataavailable = (ev) => {
          if (ev.data.size) recordChunksRef.current.push(ev.data);
        };
        mr.onstop = () => {
          stream.getTracks().forEach((t) => t.stop());
          const blob = new Blob(recordChunksRef.current, { type: mr.mimeType || "audio/webm" });
          setVoiceRefFile(new File([blob], "reference-recording.webm", { type: blob.type }));
          mediaRecorderRef.current = null;
          setRecording(false);
        };
        mr.start();
        setRecording(true);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  };

  const stopRecordRef = () => {
    mediaRecorderRef.current?.stop();
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

    const base = normalizedBase();

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
      let audioFile = await textToSpeechFile(ttsText);
      try {
        const dur = await getDecodedAudioDurationSeconds(audioFile);
        setTtsDurationLine(`TTS duration: ${formatTtsDuration(dur)} (this run)`);
        setInfo(`Driving audio ${formatTtsDuration(dur)} — preparing job…`);
      } catch {
        setTtsDurationLine(null);
        setInfo("Preparing job…");
      }

      if (useVoiceClone) {
        setInfo("Cloning driving audio to target voice…");
        const ttsPcm = await audioToMonoPcm16Base64(audioFile);
        const outB64 = await cloneVoiceRest(base, bearerToken, ttsPcm, ac.signal);
        const wavBlob = pcm16Base64ToWavBlob(outB64);
        audioFile = new File([wavBlob], "driving-cloned.wav", { type: "audio/wav" });
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
      triggerBlobDownload(blob, "lipsync-output.mp4");

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
        (same as <code>test.py</code>). Standard jobs use <code>POST /api/job</code> only.
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
            id="rtdep"
            type="checkbox"
            checked={false}
            disabled
            readOnly
          />
          <label htmlFor="rtdep" className="dimmed">
            Realtime mode (deprecated — use <code>POST /api/job</code> only)
          </label>
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
            <option value="quarter">25% (quarter)</option>
            <option value="eighth">12.5% (eighth)</option>
            <option value="sixteenth">1.5625% (sixteenth)</option>
          </select>
          <p className="hint">Applies to standard <code>/api/job</code> (server upscales to full video size).</p>
        </div>

        <div className="field">
          <div className="row">
            <input
              id="vc"
              type="checkbox"
              checked={useVoiceClone}
              onChange={(e) => setUseVoiceClone(e.target.checked)}
              disabled={busy}
            />
            <label htmlFor="vc">Use voice clone (OpenVoice on MuseTalk at /api/voice)</label>
          </div>
          <p className="hint">
            Requires server <code>ENABLE_VOICE_CLONER=on</code> and <code>JWT_SECRET</code>. Train once
            per voice, then each run clones Puter TTS to that voice before lipsync.
          </p>
          {useVoiceClone ? (
            <div className="stack" style={{ marginTop: 8 }}>
              <label htmlFor="vref">Reference voice (audio file)</label>
              <input
                id="vref"
                type="file"
                accept="audio/*"
                disabled={busy || voiceBusy}
                onChange={(e) => setVoiceRefFile(e.target.files?.[0] ?? null)}
              />
              <div className="row">
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={recording ? stopRecordRef : startRecordRef}
                  disabled={busy || voiceBusy}
                >
                  {recording ? "Stop recording" : "Record reference"}
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => void onRegisterVoice()}
                  disabled={busy || voiceBusy}
                >
                  {voiceBusy ? "Training…" : "Register voice (train)"}
                </button>
              </div>
              {trainedVoiceId ? (
                <p className="hint">
                  <code>trained_voice_id</code>: {trainedVoiceId}{" "}
                  <button
                    type="button"
                    className="btn btn-ghost"
                    onClick={() => void navigator.clipboard.writeText(trainedVoiceId)}
                  >
                    Copy
                  </button>
                </p>
              ) : (
                <p className="hint">After train, the voice id appears here for your records.</p>
              )}
            </div>
          ) : null}
        </div>

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
