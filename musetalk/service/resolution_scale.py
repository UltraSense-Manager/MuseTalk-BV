"""Job resolution presets: process at reduced size, upscale to full before final output."""

from __future__ import annotations

import glob
import os
import subprocess
import cv2
import imageio.v2 as imageio


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
    if key in ("eighth", "12.5", "0.125"):
        return 0.125
    if key in ("lowest", "16th", "0.0625"):
        return 0.0625
    raise ValueError(
        f"invalid resolution_scale {name!r}; "
        "use one of: full, half, eighth, lowest (aliases: 100, 50, 12.5)"
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
    first = imageio.imread(files[0])
    full_h, full_w = first.shape[:2]
    new_w = max(2, int(round(full_w * scale)))
    new_h = max(2, int(round(full_h * scale)))
    for p in files:
        img = imageio.imread(p)
        small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        imageio.imwrite(p, small)
    return (full_w, full_h)


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
    vf = f"scale={width}:{height}:flags=lanczos"
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
