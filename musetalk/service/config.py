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
    )
