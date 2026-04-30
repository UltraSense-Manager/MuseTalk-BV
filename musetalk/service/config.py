import os
from dataclasses import dataclass


def _env_truthy(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or not v.strip():
        return default
    try:
        return int(v.strip())
    except ValueError:
        return default


@dataclass(frozen=True)
class ServiceConfig:
    bearer_token: str | None
    jwt_secret: str | None
    jwt_algorithm: str
    secured_mode: bool
    gradio_username: str | None
    gradio_password: str | None
    gpu_multipart_field: str
    cpu_workers: int
    enable_parallel_blend: bool
    enable_parallel_audio_frame_overlap: bool
    enable_parallel_realtime_prep: bool
    enable_streaming_standard: bool
    enable_streaming_realtime: bool
    streaming_pipe_buffer_frames: int
    ffmpeg_video_encoder: str
    ffmpeg_encoder_preset: str
    ffmpeg_encoder_crf: str
    ffmpeg_encoder_cq: str
    ffmpeg_use_gpu_scale: bool
    standard_batch_size: int
    realtime_batch_size_default: int
    landmark_batch_size: int
    enable_voice_cloner: bool


def load_service_config() -> ServiceConfig:
    token = os.environ.get("BEARER_TOKEN", "").strip()
    jwt_secret = os.environ.get("JWT_SECRET", "").strip()
    jwt_algorithm = os.environ.get("JWT_ALGORITHM", "HS256").strip() or "HS256"
    field = os.environ.get("GPU_MULTIPART_FIELD", "file").strip() or "file"
    cpu_workers = max(1, _env_int("CPU_WORKERS", 2))

    secured = _env_truthy("SECURED_MODE", default=False)
    enable_parallel_blend = _env_truthy("ENABLE_PARALLEL_BLEND", default=False)
    enable_parallel_audio_frame_overlap = _env_truthy(
        "ENABLE_PARALLEL_AUDIO_FRAME_OVERLAP", default=True
    )
    enable_parallel_realtime_prep = _env_truthy(
        "ENABLE_PARALLEL_REALTIME_PREP", default=False
    )
    enable_streaming_standard = _env_truthy(
        "ENABLE_STREAMING_STANDARD", default=False
    )
    enable_streaming_realtime = _env_truthy(
        "ENABLE_STREAMING_REALTIME", default=False
    )
    streaming_pipe_buffer_frames = max(1, _env_int("STREAMING_PIPE_BUFFER_FRAMES", 4))
    ffmpeg_video_encoder = os.environ.get("FFMPEG_VIDEO_ENCODER", "h264_nvenc").strip() or "h264_nvenc"
    ffmpeg_encoder_preset = os.environ.get("FFMPEG_ENCODER_PRESET", "p5").strip() or "p5"
    ffmpeg_encoder_crf = os.environ.get("FFMPEG_ENCODER_CRF", "18").strip() or "18"
    ffmpeg_encoder_cq = os.environ.get("FFMPEG_ENCODER_CQ", "23").strip() or "23"
    ffmpeg_use_gpu_scale = _env_truthy("FFMPEG_USE_GPU_SCALE", default=True)
    standard_batch_size = max(1, _env_int("STANDARD_BATCH_SIZE", 8))
    realtime_batch_size_default = max(1, _env_int("REALTIME_BATCH_SIZE_DEFAULT", 20))
    landmark_batch_size = max(1, _env_int("LANDMARK_BATCH_SIZE", 1))
    enable_voice_cloner = _env_truthy("ENABLE_VOICE_CLONER", default=False)

    # Prefer GRADIO_* to avoid accidental use of OS USER; USER/PASS still supported.
    user = (
        os.environ.get("GRADIO_USER", "").strip()
        or os.environ.get("USER", "").strip()
        or None
    )
    password = (
        os.environ.get("GRADIO_PASS", "").strip()
        or os.environ.get("PASS", "").strip()
        or None
    )

    return ServiceConfig(
        bearer_token=token or None,
        jwt_secret=jwt_secret or None,
        jwt_algorithm=jwt_algorithm,
        secured_mode=secured,
        gradio_username=user,
        gradio_password=password,
        gpu_multipart_field=field,
        cpu_workers=cpu_workers,
        enable_parallel_blend=enable_parallel_blend,
        enable_parallel_audio_frame_overlap=enable_parallel_audio_frame_overlap,
        enable_parallel_realtime_prep=enable_parallel_realtime_prep,
        enable_streaming_standard=enable_streaming_standard,
        enable_streaming_realtime=enable_streaming_realtime,
        streaming_pipe_buffer_frames=streaming_pipe_buffer_frames,
        ffmpeg_video_encoder=ffmpeg_video_encoder,
        ffmpeg_encoder_preset=ffmpeg_encoder_preset,
        ffmpeg_encoder_crf=ffmpeg_encoder_crf,
        ffmpeg_encoder_cq=ffmpeg_encoder_cq,
        ffmpeg_use_gpu_scale=ffmpeg_use_gpu_scale,
        standard_batch_size=standard_batch_size,
        realtime_batch_size_default=realtime_batch_size_default,
        landmark_batch_size=landmark_batch_size,
        enable_voice_cloner=enable_voice_cloner,
    )
