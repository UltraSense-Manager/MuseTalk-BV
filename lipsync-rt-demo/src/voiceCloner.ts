/** Voice cloner REST client (mounted on MuseTalk at /api/voice when ENABLE_VOICE_CLONER=on). */

const PCM_SAMPLE_RATE = 16000;
const CHUNK_CHARS = 450_000;

function normalizeBase(base: string): string {
  return base.trim().replace(/\/+$/, "");
}

function authHeaders(token: string): HeadersInit {
  const t = token.trim();
  if (!t) return {};
  return { Authorization: `Bearer ${t}`, "Content-Type": "application/json" };
}

function floatTo16BitPCM(input: Float32Array): Uint8Array {
  const buf = new Uint8Array(input.length * 2);
  const view = new DataView(buf.buffer);
  for (let i = 0; i < input.length; i++) {
    let s = Math.max(-1, Math.min(1, input[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buf;
}

/** Decode audio file/blob to mono PCM16 @ 16kHz, return base64 of raw PCM bytes. */
export async function audioToMonoPcm16Base64(blob: Blob): Promise<string> {
  const raw = await blob.arrayBuffer();
  const ctx = new AudioContext();
  let decoded: AudioBuffer;
  try {
    decoded = await ctx.decodeAudioData(raw.slice(0));
  } finally {
    await ctx.close();
  }
  const frames = Math.min(
    Math.ceil(decoded.duration * PCM_SAMPLE_RATE),
    PCM_SAMPLE_RATE * 20
  );
  const oc = new OfflineAudioContext(
    decoded.numberOfChannels,
    frames,
    PCM_SAMPLE_RATE
  );
  const src = oc.createBufferSource();
  src.buffer = decoded;
  src.connect(oc.destination);
  src.start(0);
  const rendered = await oc.startRendering();
  const mono = new Float32Array(rendered.length);
  for (let c = 0; c < rendered.numberOfChannels; c++) {
    const d = rendered.getChannelData(c);
    for (let i = 0; i < mono.length; i++) mono[i] += d[i] / rendered.numberOfChannels;
  }
  const pcm = floatTo16BitPCM(mono);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < pcm.length; i += chunk) {
    binary += String.fromCharCode.apply(null, pcm.subarray(i, i + chunk) as unknown as number[]);
  }
  return btoa(binary);
}

function writeString(view: DataView, offset: number, s: string) {
  for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
}

/** Wrap raw PCM16 mono LE into a WAV Blob (16 kHz). */
export function pcm16Base64ToWavBlob(pcmBase64: string): Blob {
  const binary = atob(pcmBase64);
  const pcmLen = binary.length;
  const buffer = new ArrayBuffer(44 + pcmLen);
  const view = new DataView(buffer);
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + pcmLen, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, PCM_SAMPLE_RATE, true);
  view.setUint32(28, PCM_SAMPLE_RATE * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, pcmLen, true);
  const out = new Uint8Array(buffer);
  for (let i = 0; i < pcmLen; i++) out[44 + i] = binary.charCodeAt(i);
  return new Blob([out], { type: "audio/wav" });
}

async function postJson(
  base: string,
  token: string,
  path: string,
  body: object,
  signal?: AbortSignal
): Promise<unknown> {
  const r = await fetch(`${normalizeBase(base)}${path}`, {
    method: "POST",
    headers: authHeaders(token),
    body: JSON.stringify(body),
    signal,
  });
  const text = await r.text();
  if (r.status === 401) throw new Error("401 Unauthorized — check bearer token.");
  if (!r.ok) throw new Error(`Voice cloner ${path} failed (${r.status}): ${text.slice(0, 400)}`);
  return JSON.parse(text) as unknown;
}

/** POST /api/voice/train start → reference chunks → end. Returns trained_voice_id. */
export async function trainVoiceRest(
  museTalkBase: string,
  token: string,
  pcmBase64: string,
  signal?: AbortSignal
): Promise<string> {
  const prefix = "/api/voice";
  await postJson(museTalkBase, token, `${prefix}/train`, { operation: "start" }, signal);
  for (let i = 0; i < pcmBase64.length; i += CHUNK_CHARS) {
    const part = pcmBase64.slice(i, i + CHUNK_CHARS);
    await postJson(museTalkBase, token, `${prefix}/train`, { reference: part }, signal);
  }
  const end = (await postJson(
    museTalkBase,
    token,
    `${prefix}/train`,
    { operation: "end" },
    signal
  )) as { trained_voice_id?: string };
  const id = end.trained_voice_id;
  if (!id) throw new Error("train end: missing trained_voice_id");
  return id;
}

/** POST /api/voice/clone — returns base64 PCM of cloned audio. */
export async function cloneVoiceRest(
  museTalkBase: string,
  token: string,
  pcmBase64: string,
  signal?: AbortSignal
): Promise<string> {
  const out = (await postJson(
    museTalkBase,
    token,
    "/api/voice/clone",
    { base: pcmBase64 },
    signal
  )) as { output_path?: string };
  const b64 = out.output_path;
  if (!b64) throw new Error("clone: missing output_path");
  return b64;
}

export async function voiceHealth(
  museTalkBase: string,
  signal?: AbortSignal
): Promise<{ voice_cloner?: boolean }> {
  const r = await fetch(`${normalizeBase(museTalkBase)}/api/health`, { signal });
  if (!r.ok) return {};
  return (await r.json()) as { voice_cloner?: boolean };
}
