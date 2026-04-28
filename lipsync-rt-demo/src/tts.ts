/** Puter.js TTS → File for multipart upload (WAV). */

const MAX_CHARS = 3000;

export function validateTtsText(text: string): string | null {
  const t = text.trim();
  if (!t) return "Enter text for speech.";
  if (t.length > MAX_CHARS) return `Text must be at most ${MAX_CHARS} characters.`;
  return null;
}

export function puterAvailable(): boolean {
  return typeof window !== "undefined" && !!window.puter?.ai?.txt2speech;
}

export async function textToSpeechFile(text: string): Promise<File> {
  if (!window.puter?.ai?.txt2speech) {
    throw new Error("Puter.js is not loaded (script https://js.puter.com/v2/).");
  }
  const el = await window.puter.ai.txt2speech(text.trim(), {
    response_format: "wav",
  });
  const src = el.src;
  if (!src) throw new Error("TTS returned no audio URL.");
  const res = await fetch(src);
  if (!res.ok) throw new Error(`Could not fetch synthesized audio (${res.status}).`);
  const blob = await res.blob();
  const type = blob.type || "audio/wav";
  return new File([blob], "tts.wav", { type });
}
