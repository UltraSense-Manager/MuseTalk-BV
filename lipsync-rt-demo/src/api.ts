/** Mirrors test.py: multipart submit, poll /api/job/{id}, download. */

export type JobSubmitResponse = {
  job_id: string;
  status?: string;
  kind?: string;
  avatar_id?: string;
  realtime_prep_frames?: number;
  reuse_avatar?: boolean;
};

export type JobStatusBody = {
  job_id: string;
  status: string;
  message?: string;
  kind?: string;
  avatar_id?: string;
};

export function authHeaders(token: string): HeadersInit {
  const t = token.trim();
  if (!t) return {};
  return { Authorization: `Bearer ${t}` };
}

function normalizeBase(base: string): string {
  return base.trim().replace(/\/+$/, "");
}

export async function submitStandardJob(
  base: string,
  token: string,
  audio: File,
  video: File,
  form: Record<string, string>,
  signal?: AbortSignal
): Promise<JobSubmitResponse> {
  const fd = new FormData();
  fd.append("audio", audio, audio.name);
  fd.append("video", video, video.name);
  for (const [k, v] of Object.entries(form)) fd.append(k, v);

  const r = await fetch(`${normalizeBase(base)}/api/job`, {
    method: "POST",
    headers: authHeaders(token),
    body: fd,
    signal,
  });
  const text = await r.text();
  if (r.status === 401) throw new Error("401 Unauthorized — check bearer token.");
  if (!r.ok) throw new Error(`Submit failed (${r.status}): ${text.slice(0, 400)}`);
  let payload: JobSubmitResponse;
  try {
    payload = JSON.parse(text) as JobSubmitResponse;
  } catch {
    throw new Error("Submit response is not JSON");
  }
  if (!payload.job_id) throw new Error("No job_id in response");
  return payload;
}

export async function submitRealtimeJob(
  base: string,
  token: string,
  audio: File,
  form: Record<string, string>,
  video: File | null,
  signal?: AbortSignal
): Promise<JobSubmitResponse> {
  const fd = new FormData();
  fd.append("audio", audio, audio.name);
  if (video) fd.append("video", video, video.name);
  for (const [k, v] of Object.entries(form)) fd.append(k, v);

  const r = await fetch(`${normalizeBase(base)}/api/realtime/job`, {
    method: "POST",
    headers: authHeaders(token),
    body: fd,
    signal,
  });
  const text = await r.text();
  if (r.status === 401) throw new Error("401 Unauthorized — check bearer token.");
  if (!r.ok) throw new Error(`Submit failed (${r.status}): ${text.slice(0, 400)}`);
  let payload: JobSubmitResponse;
  try {
    payload = JSON.parse(text) as JobSubmitResponse;
  } catch {
    throw new Error("Submit response is not JSON");
  }
  if (!payload.job_id) throw new Error("No job_id in response");
  return payload;
}

export async function pollJob(
  base: string,
  token: string,
  jobId: string,
  signal?: AbortSignal
): Promise<JobStatusBody> {
  const r = await fetch(`${normalizeBase(base)}/api/job/${encodeURIComponent(jobId)}`, {
    headers: { ...authHeaders(token) },
    signal,
  });
  const text = await r.text();
  if (r.status !== 200) throw new Error(`Poll HTTP ${r.status}: ${text.slice(0, 400)}`);
  return JSON.parse(text) as JobStatusBody;
}

export async function downloadJob(
  base: string,
  token: string,
  jobId: string,
  signal?: AbortSignal
): Promise<Blob> {
  const r = await fetch(
    `${normalizeBase(base)}/api/job/${encodeURIComponent(jobId)}/download`,
    {
      headers: { ...authHeaders(token) },
      signal,
    }
  );
  if (!r.ok) {
    const text = await r.text();
    throw new Error(`Download HTTP ${r.status}: ${text.slice(0, 400)}`);
  }
  return r.blob();
}

export async function pollUntilDone(
  base: string,
  token: string,
  jobId: string,
  opts: {
    intervalMs: number;
    deadlineMs: number;
    signal?: AbortSignal;
    onTick?: (body: JobStatusBody) => void;
  }
): Promise<JobStatusBody> {
  const start = performance.now();
  for (;;) {
    if (opts.signal?.aborted) throw new DOMException("Aborted", "AbortError");
    if (performance.now() - start > opts.deadlineMs) {
      throw new Error("Timed out waiting for job");
    }
    const body = await pollJob(base, token, jobId, opts.signal);
    opts.onTick?.(body);
    if (body.status === "done") return body;
    if (body.status === "error") {
      throw new Error(body.message || "Job failed");
    }
    await new Promise((r) => setTimeout(r, opts.intervalMs));
  }
}
