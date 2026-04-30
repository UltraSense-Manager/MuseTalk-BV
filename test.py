#!/usr/bin/env python3
"""
End-to-end checks for the MuseTalk HTTP API.

Modes:
  standard (default) — POST /api/job, poll, download
  realtime           — POST /api/realtime/job (first N frames prep or clone reuse), poll, download

Requires: requests, test audio + video (see --audio / --video).

  python test.py http://127.0.0.1:7860
  python test.py http://127.0.0.1:7860 --mode realtime --out realtime_out.mp4
  python test.py http://127.0.0.1:7860 --token "$BEARER_TOKEN"
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests


def _resolve_path(p: str) -> str:
    if os.path.isfile(p):
        return os.path.abspath(p)
    beside_script = Path(__file__).resolve().parent / p
    if beside_script.is_file():
        return str(beside_script.resolve())
    return p


def _guess_video_mime(video_path: str) -> str:
    mt, _ = mimetypes.guess_type(video_path)
    if mt and mt.startswith("video/"):
        return mt
    ext = os.path.splitext(video_path)[1].lower()
    return {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
    }.get(ext, "application/octet-stream")


def _guess_audio_mime(audio_path: str) -> str:
    mt, _ = mimetypes.guess_type(audio_path)
    if mt and mt.startswith("audio/"):
        return mt
    ext = os.path.splitext(audio_path)[1].lower()
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
    }.get(ext, "application/octet-stream")


def _auth_headers(token: str) -> dict[str, str]:
    if not token.strip():
        return {}
    return {"Authorization": f"Bearer {token.strip()}"}


def _submit_multipart_job(
    base: str,
    headers: dict[str, str],
    audio_path: str,
    video_path: Optional[str],
    submit_path: str,
    form_data: dict[str, str],
    submit_timeout: int,
) -> tuple[int, requests.Response | None]:
    url = f"{base}{submit_path}"
    print(f"POST {url}")
    try:
        af = open(audio_path, "rb")
        try:
            files: dict[str, tuple[str, object, str]] = {
                "audio": (
                    os.path.basename(audio_path),
                    af,
                    _guess_audio_mime(audio_path),
                ),
            }
            if video_path:
                vf = open(video_path, "rb")
                try:
                    files["video"] = (
                        os.path.basename(video_path),
                        vf,
                        _guess_video_mime(video_path),
                    )
                    r = requests.post(
                        url,
                        headers=headers,
                        files=files,
                        data=form_data,
                        timeout=submit_timeout,
                    )
                finally:
                    vf.close()
            else:
                r = requests.post(
                    url,
                    headers=headers,
                    files=files,
                    data=form_data,
                    timeout=submit_timeout,
                )
        finally:
            af.close()
    except requests.RequestException as e:
        print(f"error: submit failed: {e}", file=sys.stderr)
        return 1, None
    return 0, r


def _handle_submit_response(r: requests.Response) -> tuple[int, str | None]:
    if r.status_code == 401:
        print(
            "error: 401 Unauthorized — set BEARER_TOKEN or --token to match the server.",
            file=sys.stderr,
        )
        return 1, None
    if not r.ok:
        print(f"error: submit HTTP {r.status_code}: {r.text[:500]}", file=sys.stderr)
        return 1, None
    try:
        payload = r.json()
    except ValueError:
        print(f"error: submit response is not JSON: {r.text[:300]}", file=sys.stderr)
        return 1, None
    job_id = payload.get("job_id")
    if not job_id:
        print(f"error: no job_id in response: {payload}", file=sys.stderr)
        return 1, None
    extra = f" kind={payload.get('kind', '')!r}"
    clone = payload.get("clone_id") or payload.get("user_id")
    if clone:
        extra += f" clone_id={clone!r}"
    print(f"job_id={job_id} status={payload.get('status', '')}{extra}")
    return 0, job_id


def _poll_until_done(
    base: str,
    headers: dict[str, str],
    job_id: str,
    deadline: float,
    poll_interval: float,
) -> int:
    while time.monotonic() < deadline:
        try:
            st = requests.get(
                f"{base}/api/job/{job_id}",
                headers=headers,
                timeout=60,
            )
        except requests.RequestException as e:
            print(f"error: status poll failed: {e}", file=sys.stderr)
            return 1

        if st.status_code != 200:
            print(
                f"error: status HTTP {st.status_code}: {st.text[:500]}",
                file=sys.stderr,
            )
            return 1

        try:
            body = st.json()
        except ValueError:
            print(f"error: status not JSON: {st.text[:300]}", file=sys.stderr)
            return 1

        status = body.get("status", "")
        if status == "done":
            return 0
        if status == "error":
            print(f"error: job failed: {body.get('message', body)}", file=sys.stderr)
            return 1

        av = body.get("clone_id") or body.get("user_id")
        avs = f" clone_id={av!r}" if av else ""
        print(f"  status={status!r} kind={body.get('kind', '')!r}{avs} …")
        time.sleep(poll_interval)
    print("error: timed out waiting for job", file=sys.stderr)
    return 1


def _download_result(
    base: str,
    headers: dict[str, str],
    job_id: str,
    out_arg: str,
    deadline: float,
) -> tuple[int, int]:
    print(f"GET {base}/api/job/{job_id}/download -> {out_arg}")
    try:
        dl = requests.get(
            f"{base}/api/job/{job_id}/download",
            headers=headers,
            timeout=min(600, max(60, int(deadline - time.monotonic()))),
        )
    except requests.RequestException as e:
        print(f"error: download failed: {e}", file=sys.stderr)
        return 1, 0

    if not dl.ok:
        print(
            f"error: download HTTP {dl.status_code}: {dl.text[:500]}",
            file=sys.stderr,
        )
        return 1, 0

    out_path = os.path.abspath(out_arg)
    with open(out_path, "wb") as out:
        out.write(dl.content)
    return 0, len(dl.content)


def run_standard_job(
    base: str,
    headers: dict[str, str],
    audio_path: str,
    video_path: str,
    out: str,
    deadline: float,
    poll_interval: float,
    job_started: float,
    form: dict[str, str],
) -> int:
    submit_timeout = min(300, max(30, int(deadline - time.monotonic())))
    err, r = _submit_multipart_job(
        base,
        headers,
        audio_path,
        video_path,
        "/api/job",
        form.copy(),
        submit_timeout,
    )
    if err or r is None:
        return 1

    err, job_id = _handle_submit_response(r)
    if err or not job_id:
        return 1

    if _poll_until_done(base, headers, job_id, deadline, poll_interval) != 0:
        return 1

    err, nbytes = _download_result(base, headers, job_id, out, deadline)
    if err:
        return 1

    elapsed = time.monotonic() - job_started
    print(f"wrote {os.path.abspath(out)} ({nbytes} bytes)")
    print(f"elapsed {elapsed:.1f}s")
    return 0


def run_realtime_job(
    base: str,
    headers: dict[str, str],
    audio_path: str,
    video_path: Optional[str],
    out: str,
    deadline: float,
    poll_interval: float,
    job_started: float,
    form: dict[str, str],
) -> int:
    submit_timeout = min(300, max(30, int(deadline - time.monotonic())))
    err, r = _submit_multipart_job(
        base,
        headers,
        audio_path,
        video_path,
        "/api/realtime/job",
        form.copy(),
        submit_timeout,
    )
    if err or r is None:
        return 1

    err, job_id = _handle_submit_response(r)
    if err or not job_id:
        return 1

    if _poll_until_done(base, headers, job_id, deadline, poll_interval) != 0:
        return 1

    err, nbytes = _download_result(base, headers, job_id, out, deadline)
    if err:
        return 1

    elapsed = time.monotonic() - job_started
    print(f"wrote {os.path.abspath(out)} ({nbytes} bytes)")
    print(f"elapsed {elapsed:.1f}s (realtime)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test MuseTalk /api/job or /api/realtime/job + download."
    )
    parser.add_argument(
        "base_url",
        help="API base URL, e.g. http://127.0.0.1:7860",
    )
    parser.add_argument(
        "--mode",
        choices=("standard", "realtime"),
        default="standard",
        help="standard = POST /api/job; realtime = POST /api/realtime/job",
    )
    parser.add_argument(
        "--audio",
        default="test_audio.wav",
        help="Driving audio file",
    )
    parser.add_argument(
        "--video",
        default="test_video.mov",
        help="Reference video file",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output MP4 path (default: test_job_output.mp4 or test_realtime_output.mp4)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("BEARER_TOKEN", ""),
        help="Bearer token (default: env BEARER_TOKEN)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between status polls",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Max total seconds for submit + poll + download",
    )
    # Realtime-only tuning (ignored for standard mode)
    parser.add_argument(
        "--realtime-prep-frames",
        type=int,
        default=30,
        help="First N video frames for realtime prep (realtime mode only)",
    )
    parser.add_argument(
        "--realtime-batch-size",
        type=int,
        default=20,
        help="Realtime inference batch size (realtime mode only)",
    )
    parser.add_argument(
        "--realtime-fps",
        type=int,
        default=25,
        help="FPS for realtime mux (realtime mode only)",
    )
    parser.add_argument(
        "--bbox-shift",
        type=float,
        default=0,
        help="bbox_shift form field",
    )
    parser.add_argument(
        "--extra-margin",
        type=int,
        default=10,
        help="extra_margin form field",
    )
    parser.add_argument(
        "--parsing-mode",
        choices=("jaw", "raw"),
        default="jaw",
        help="parsing_mode form field",
    )
    parser.add_argument(
        "--left-cheek-width",
        type=int,
        default=90,
        help="left_cheek_width form field",
    )
    parser.add_argument(
        "--right-cheek-width",
        type=int,
        default=90,
        help="right_cheek_width form field",
    )
    parser.add_argument(
        "--clone-id",
        default="",
        help="Realtime: POST clone_id (optional; if unset server can infer from JWT uid/sub)",
    )
    parser.add_argument(
        "--use-clone",
        action="store_true",
        help="Realtime: POST use_clone=true to skip video and reuse persisted clone materials",
    )
    parser.add_argument(
        "--resolution-scale",
        default="full",
        choices=("full", "half", "quarter", "eighth", "sixteenth", "50", "25", "12.5", "1.5625"),
        help="Process at reduced resolution then upscale to full (standard + realtime)",
    )

    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    audio_path = _resolve_path(args.audio)
    clone_id = ""
    if args.mode == "realtime":
        clone_id = (args.clone_id or "").strip()
    video_path: Optional[str] = None
    if not args.use_clone:
        video_path = _resolve_path(args.video)
    out = args.out or (
        "test_realtime_output.mp4"
        if args.mode == "realtime"
        else "test_job_output.mp4"
    )

    if not os.path.isfile(audio_path):
        print(f"error: missing audio file: {audio_path}", file=sys.stderr)
        return 1
    if not args.use_clone:
        if not video_path or not os.path.isfile(video_path):
            print(
                f"error: missing video file: {video_path or args.video}",
                file=sys.stderr,
            )
            return 1

    headers = _auth_headers(args.token)
    deadline = time.monotonic() + args.timeout
    job_started = time.monotonic()

    common_form = {
        "bbox_shift": str(args.bbox_shift),
        "extra_margin": str(args.extra_margin),
        "parsing_mode": args.parsing_mode,
        "left_cheek_width": str(args.left_cheek_width),
        "right_cheek_width": str(args.right_cheek_width),
        "resolution_scale": str(args.resolution_scale),
    }

    if args.mode == "standard":
        return run_standard_job(
            base,
            headers,
            audio_path,
            video_path,
            out,
            deadline,
            args.poll_interval,
            job_started,
            common_form,
        )

    rt_form = {
        **common_form,
        "realtime_prep_frames": str(args.realtime_prep_frames),
        "realtime_batch_size": str(args.realtime_batch_size),
        "realtime_fps": str(args.realtime_fps),
    }
    rt_form["use_clone"] = "true" if args.use_clone else "false"
    if clone_id:
        rt_form["clone_id"] = clone_id
    return run_realtime_job(
        base,
        headers,
        audio_path,
        video_path,
        out,
        deadline,
        args.poll_interval,
        job_started,
        rt_form,
    )


if __name__ == "__main__":
    raise SystemExit(main())
