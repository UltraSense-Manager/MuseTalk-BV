"""Job resolution presets: process at reduced size, upscale to full before final output."""

from __future__ import annotations

import glob
import os
import subprocess


def _even_dim(v: int) -> int:
    iv = int(v)
    if iv < 2:
        return 2
    return iv if iv % 2 == 0 else iv - 1


def parse_resolution_scale(name: str) -> float:
    """
    Map API / CLI string to a linear scale in (0, 1].

    Presets:
      full / 100 — 1.0
      half / 50 — 0.5
      eighth / 12.5 — 0.125
      lowest — 0.0625 (1/16 of linear dimensions)
    """
    key = (name or "full").strip().lower().replace(" ", "").replace("%", "")
    if key in ("full", "100", "1", "1.0"):
        return 1.0
    if key in ("half", "50", "0.5"):
        return 0.5
    if key in ("quarter", "25", "0.25"):
        return 0.25
    if key in ("eighth", "12.5", "0.125"):
        return 0.125
    if key in ("sixteenth", "1.5625", "0.0625"):
        return 0.0625
    raise ValueError(
        f"invalid resolution_scale {name!r}; "
        "use one of: full, half, quarter, eighth, lowest (aliases: 100, 50, 25, 12.5, 1.5625)"
    )


def downscale_png_dir_inplace(
    dir_path: str, scale: float
) -> tuple[int, int] | None:
    """
    If scale < 1, resize every *.png in dir_path in place.
    Returns (full_w, full_h) from the first frame before downscale, or None if scale>=1.
    """
    if scale >= 1.0 - 1e-9:
        return None
    files = sorted(glob.glob(os.path.join(dir_path, "*.png")))
    if not files:
        raise RuntimeError(f"no PNG frames under {dir_path!r}")
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            files[0],
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0 or "x" not in probe.stdout:
        raise RuntimeError(
            f"ffprobe failed on first frame ({probe.returncode}): {probe.stderr[-1200:]!r}"
        )
    full_w, full_h = [int(v) for v in probe.stdout.strip().split("x", 1)]
    new_w = _even_dim(int(round(full_w * scale)))
    new_h = _even_dim(int(round(full_h * scale)))
    tmp_dir = os.path.join(dir_path, "_scaled_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "warning",
            "-start_number",
            "0",
            "-i",
            os.path.join(dir_path, "%08d.png"),
            "-vf",
            f"scale={new_w}:{new_h}:flags=area",
            "-start_number",
            "0",
            os.path.join(tmp_dir, "%08d.png"),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg downscale failed ({proc.returncode}): {proc.stderr[-1500:]!r}"
        )
    scaled = sorted(glob.glob(os.path.join(tmp_dir, "*.png")))
    if not scaled:
        raise RuntimeError("ffmpeg downscale produced no frames")
    for p in files:
        os.remove(p)
    for p in scaled:
        os.replace(p, os.path.join(dir_path, os.path.basename(p)))
    os.rmdir(tmp_dir)
    if not os.path.exists(os.path.join(dir_path, "00000000.png")):
        raise RuntimeError(
            "downscale output missing 00000000.png; frame numbering was not preserved"
        )
    return (full_w, full_h)


def upscale_video_stream(
    input_mp4: str,
    width: int,
    height: int,
    out_mp4: str,
) -> None:
    """Upscale video stream only (no audio) to WxH."""
    even_w = _even_dim(width)
    even_h = _even_dim(height)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "warning",
            "-i",
            input_mp4,
            "-vf",
            f"scale={even_w}:{even_h}:flags=lanczos",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            out_mp4,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg upscale video-only failed ({proc.returncode}): {proc.stderr[-1500:]!r}"
        )


def upscale_video_replace_audio(
    scaled_video_with_audio: str,
    audio_path: str,
    width: int,
    height: int,
    out_mp4: str,
) -> None:
    """
    Upscale video stream to WxH, then mux original driving audio (one ffmpeg graph).
    """
    even_w = _even_dim(width)
    even_h = _even_dim(height)
    vf = f"scale={even_w}:{even_h}:flags=lanczos"
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "warning",
            "-i",
            scaled_video_with_audio,
            "-i",
            audio_path,
            "-filter_complex",
            f"[0:v]{vf}[v]",
            "-map",
            "[v]",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            out_mp4,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg upscale+audio failed ({proc.returncode}): {proc.stderr[-1500:]!r}"
        )
