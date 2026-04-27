import os
from dataclasses import dataclass


def _env_truthy(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ServiceConfig:
    bearer_token: str | None
    secured_mode: bool
    gradio_username: str | None
    gradio_password: str | None
    gpu_multipart_field: str


def load_service_config() -> ServiceConfig:
    token = os.environ.get("BEARER_TOKEN", "").strip()
    field = os.environ.get("GPU_MULTIPART_FIELD", "file").strip() or "file"

    secured = _env_truthy("SECURED_MODE", default=False)

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
        secured_mode=secured,
        gradio_username=user,
        gradio_password=password,
        gpu_multipart_field=field,
    )
