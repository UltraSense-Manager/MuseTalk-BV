/** Puter.js TTS → File for multipart upload (WAV). */

const MAX_CHARS = 3000;

/** Human-readable duration for TTS / short clips. */
export function formatTtsDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "—";
  const decimals = seconds >= 100 ? 1 : 2;
  return `${seconds.toFixed(decimals)}s`;
}

/** Wait until `HTMLAudioElement.duration` is known (Puter TTS element). */
export function waitAudioElementDuration(
  el: HTMLAudioElement,
  timeoutMs = 15000
): Promise<number> {
  return new Promise((resolve, reject) => {
    let settled = false;
    const tryResolve = () => {
      if (settled) return;
      const d = el.duration;
      if (Number.isFinite(d) && d > 0 && !Number.isNaN(d)) {
        settled = true;
        cleanup();
        resolve(d);
      }
    };
    const fail = (err: Error) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(err);
    };
    const cleanup = () => {
      window.clearTimeout(to);
      el.removeEventListener("loadedmetadata", tryResolve);
      el.removeEventListener("durationchange", tryResolve);
      el.removeEventListener("error", onErr);
    };
    const onErr = () => fail(new Error("audio element error"));
    const to = window.setTimeout(
      () => fail(new Error("timeout waiting for TTS duration metadata")),
      timeoutMs
    );
    el.addEventListener("loadedmetadata", tryResolve);
    el.addEventListener("durationchange", tryResolve);
    el.addEventListener("error", onErr);
    tryResolve();
  });
}

/** Decode uploaded/synthesized audio file to read exact duration (WAV from TTS). */
export async function getDecodedAudioDurationSeconds(file: File): Promise<number> {
  const raw = await file.arrayBuffer();
  const ctx = new AudioContext();
  try {
    const buf = await ctx.decodeAudioData(raw.slice(0));
    return buf.duration;
  } finally {
    await ctx.close();
  }
}

export function validateTtsText(text: string): string | null {
  const t = text.trim();
  if (!t) return "Enter text for speech.";
  if (t.length > MAX_CHARS) return `Text must be at most ${MAX_CHARS} characters.`;
  return null;
}

export function puterAvailable(): boolean {
  return typeof window !== "undefined" && !!window.puter?.ai?.txt2speech;
}

/** Same synthesis as upload path; returns the audio element for playback. */
export async function previewTextToSpeech(text: string): Promise<HTMLAudioElement> {
  if (!window.puter?.ai?.txt2speech) {
    throw new Error("Puter.js is not loaded (script https://js.puter.com/v2/).");
  }
  return window.puter.ai.txt2speech(text.trim(), {
    response_format: "wav",
  });
}

export async function textToSpeechFile(text: string): Promise<File> {
  if (!window.puter?.ai?.txt2speech) {
    throw new Error("Puter.js is not loaded (script https://js.puter.com/v2/).");
  }
  const el = await previewTextToSpeech(text);
  const src = el.src;
  if (!src) throw new Error("TTS returned no audio URL.");
  const res = await fetch(src);
  if (!res.ok) throw new Error(`Could not fetch synthesized audio (${res.status}).`);
  const blob = await res.blob();
  const type = blob.type || "audio/wav";
  return new File([blob], "tts.wav", { type });
}
