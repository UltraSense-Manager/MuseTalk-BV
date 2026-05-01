"""FFmpeg subprocess helpers: raw video on stdin -> H.264 MP4, mux with audio."""

from __future__ import annotations

import subprocess
from functools import lru_cache
from typing import IO

import cv2
import numpy as np


def even_dim(v: int) -> int:
    iv = int(v)
    if iv < 2:
        return 2
    return iv if iv % 2 == 0 else iv - 1


class FFmpegRawVideoWriter:
    """
    Stream BGR uint8 frames (HxWx3) to libx264 via rawvideo rgb24 stdin.
    """

    def __init__(
        self,
        out_mp4: str,
        width: int,
        height: int,
        fps: float = 25.0,
        codec: str = "h264_nvenc",
        preset: str = "p5",
        crf: str = "18",
        cq: str = "23",
        scale_to: tuple[int, int] | None = None,
        use_gpu_scale: bool = True,
        bufsize: int = 0,
    ) -> None:
        self.out_path = out_mp4
        self.w = even_dim(width)
        self.h = even_dim(height)
        self.fps = float(fps)
        self.codec = codec.strip() or "h264_nvenc"
        self.preset = preset.strip() or "p5"
        self.crf = crf.strip() or "18"
        self.cq = cq.strip() or "23"
        self.scale_to = (
            (even_dim(scale_to[0]), even_dim(scale_to[1])) if scale_to is not None else None
        )
        self.use_gpu_scale = bool(use_gpu_scale)
        self._proc: subprocess.Popen | None = None
        self._stdin: IO[bytes] | None = None
        vf_parts = []
        if self.scale_to is not None:
            tw, th = self.scale_to
            if (
                self.use_gpu_scale
                and self.codec.endswith("_nvenc")
                and has_scale_cuda_filter()
            ):
                vf_parts.append(f"scale_cuda={tw}:{th}")
            else:
                vf_parts.append(f"scale={tw}:{th}:flags=bilinear")
        vf_parts.append("format=yuv420p")
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{self.w}x{self.h}",
            "-framerate",
            str(self.fps),
            "-i",
            "-",
            "-vf",
            ",".join(vf_parts),
            "-c:v",
            self.codec,
            "-preset",
            self.preset,
        ]
        if self.codec.endswith("_nvenc"):
            cmd += [
                "-cq",
                self.cq,
            ]
        else:
            cmd += [
                "-crf",
                self.crf,
            ]
        cmd += [
            "-pix_fmt",
            "yuv420p",
            out_mp4,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=bufsize,
        )
        self._stdin = self._proc.stdin

    def write_frame_bgr(self, bgr: np.ndarray) -> None:
        if self._proc is None or self._stdin is None:
            raise RuntimeError("FFmpegRawVideoWriter is closed")
        if bgr.dtype != np.uint8:
            bgr = np.asarray(bgr, dtype=np.uint8)
        h, w = bgr.shape[:2]
        if w != self.w or h != self.h:
            bgr = cv2.resize(bgr, (self.w, self.h), interpolation=cv2.INTER_AREA)
        if not bgr.flags["C_CONTIGUOUS"]:
            bgr = np.ascontiguousarray(bgr)
        self._stdin.write(bgr.tobytes())

    def close(self) -> None:
        if self._stdin is not None:
            try:
                self._stdin.close()
            except BrokenPipeError:
                pass
            self._stdin = None
        if self._proc is not None:
            err = self._proc.stderr.read() if self._proc.stderr else b""
            rc = self._proc.wait()
            self._proc = None
            if rc != 0:
                tail = err.decode("utf-8", errors="replace")[-2000:]
                raise RuntimeError(
                    f"ffmpeg rawvideo encode failed ({rc}): {tail!r}"
                )

    def __enter__(self) -> FFmpegRawVideoWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def mux_video_with_audio(video_mp4: str, audio_path: str, out_mp4: str) -> None:
    """Mux existing H.264 video stream with driving audio (AAC)."""
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "warning",
            "-i",
            video_mp4,
            "-i",
            audio_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
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
            f"ffmpeg mux failed ({proc.returncode}): {proc.stderr[-2000:]!r}"
        )


@lru_cache(maxsize=1)
def has_nvenc_encoder() -> bool:
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return False
    if proc.returncode != 0:
        return False
    out = f"{proc.stdout}\n{proc.stderr}"
    return "h264_nvenc" in out


@lru_cache(maxsize=1)
def has_working_nvenc() -> bool:
    """
    True only when h264_nvenc is both listed and can initialize for a tiny encode.
    This avoids false positives on hosts where FFmpeg was compiled with NVENC
    support but runtime driver/device wiring is not usable.
    """
    if not has_nvenc_encoder():
        return False
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=640x480:r=1",
                "-frames:v",
                "1",
                "-c:v",
                "h264_nvenc",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return False
    return proc.returncode == 0


@lru_cache(maxsize=1)
def has_scale_cuda_filter() -> bool:
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return False
    if proc.returncode != 0:
        return False
    out = f"{proc.stdout}\n{proc.stderr}"
    return "scale_cuda" in out
