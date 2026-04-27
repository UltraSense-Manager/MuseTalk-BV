import subprocess
from pathlib import Path


def demux_muxed_mp4(muxed_path: str, out_dir: Path) -> tuple[str, str]:
    """
    Split a muxed MP4 into a silent H.264 MP4 (reference video) and a WAV (driving audio).
    Used for the worker contract POST /job uploads.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "demux_video.mp4"
    audio_path = out_dir / "demux_audio.wav"

    v_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        muxed_path,
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    a_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        muxed_path,
        "-vn",
        "-c:a",
        "pcm_s16le",
        str(audio_path),
    ]
    for cmd in (v_cmd, a_cmd):
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(
                f"ffmpeg demux failed ({p.returncode}): {p.stderr[-800:]!r}"
            )
    if not video_path.is_file() or not audio_path.is_file():
        raise RuntimeError("ffmpeg demux did not produce expected outputs")
    return str(video_path), str(audio_path)
