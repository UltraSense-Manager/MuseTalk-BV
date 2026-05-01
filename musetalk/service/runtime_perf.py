"""
In-process overrides for ServiceConfig performance fields (process-start baseline + patch).
"""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any

from musetalk.service.config import ServiceConfig

_LOCK = threading.Lock()
_BASELINE: ServiceConfig | None = None
_PATCH: dict[str, Any] = {}

# JSON / health keys -> ServiceConfig attribute
_JSON_TO_ATTR: dict[str, str] = {
    "cpu_workers": "cpu_workers",
    "parallel_blend": "enable_parallel_blend",
    "audio_frame_overlap": "enable_parallel_audio_frame_overlap",
    "parallel_realtime_prep": "enable_parallel_realtime_prep",
    "streaming_standard": "enable_streaming_standard",
    "streaming_realtime": "enable_streaming_realtime",
    "streaming_pipe_buffer_frames": "streaming_pipe_buffer_frames",
    "ffmpeg_video_encoder": "ffmpeg_video_encoder",
    "ffmpeg_encoder_preset": "ffmpeg_encoder_preset",
    "ffmpeg_encoder_crf": "ffmpeg_encoder_crf",
    "ffmpeg_encoder_cq": "ffmpeg_encoder_cq",
    "ffmpeg_use_gpu_scale": "ffmpeg_use_gpu_scale",
    "standard_batch_size": "standard_batch_size",
    "realtime_batch_size_default": "realtime_batch_size_default",
    "landmark_batch_size": "landmark_batch_size",
}

_ATTR_TO_JSON = {v: k for k, v in _JSON_TO_ATTR.items()}


def init_baseline(cfg: ServiceConfig) -> None:
    """First successful call wins (matches process-start env snapshot)."""
    global _BASELINE
    with _LOCK:
        if _BASELINE is None:
            _BASELINE = cfg


def baseline_initialized() -> bool:
    with _LOCK:
        return _BASELINE is not None


def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(int(v))
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    raise ValueError("expected boolean")


def _normalize_attr_value(attr: str, value: Any) -> Any:
    if attr == "cpu_workers":
        n = int(value)
        return max(1, min(128, n))
    if attr == "streaming_pipe_buffer_frames":
        n = int(value)
        return max(1, min(256, n))
    if attr == "standard_batch_size":
        n = int(value)
        return max(1, min(256, n))
    if attr == "realtime_batch_size_default":
        n = int(value)
        return max(1, min(128, n))
    if attr == "landmark_batch_size":
        n = int(value)
        return max(1, min(256, n))
    if attr.startswith("enable_") or attr == "ffmpeg_use_gpu_scale":
        return _coerce_bool(value)
    if attr in (
        "ffmpeg_video_encoder",
        "ffmpeg_encoder_preset",
        "ffmpeg_encoder_crf",
        "ffmpeg_encoder_cq",
    ):
        s = str(value).strip()
        if not s:
            raise ValueError("empty string")
        return s
    raise ValueError(f"unknown tunable field: {attr}")


def _effective_unlocked() -> ServiceConfig:
    assert _BASELINE is not None
    if not _PATCH:
        return _BASELINE
    return replace(_BASELINE, **_PATCH)


def get_effective_service_config() -> ServiceConfig:
    with _LOCK:
        if _BASELINE is None:
            from musetalk.service.config import load_service_config

            return load_service_config()
        return _effective_unlocked()


def _config_to_tunable_json(c: ServiceConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for attr, jkey in _ATTR_TO_JSON.items():
        out[jkey] = getattr(c, attr)
    return out


def get_state() -> dict[str, Any]:
    with _LOCK:
        if _BASELINE is None:
            return {
                "baseline": None,
                "patch": {},
                "effective": None,
            }
        base = _config_to_tunable_json(_BASELINE)
        eff = _config_to_tunable_json(_effective_unlocked())
        patch_json: dict[str, Any] = {}
        for attr in _PATCH:
            jkey = _ATTR_TO_JSON.get(attr)
            if jkey is not None:
                patch_json[jkey] = _PATCH[attr]
        return {"baseline": base, "patch": patch_json, "effective": eff}


def apply_perf_patch(body: dict[str, Any]) -> ServiceConfig:
    if not isinstance(body, dict):
        raise ValueError("body must be a JSON object")
    unknown = [k for k in body if k not in _JSON_TO_ATTR]
    if unknown:
        raise ValueError(f"unknown keys: {', '.join(unknown)}")

    updates: dict[str, Any] = {}
    with _LOCK:
        if _BASELINE is None:
            raise RuntimeError("runtime_perf baseline not initialized")
        for jkey, raw in body.items():
            if raw is None:
                continue
            attr = _JSON_TO_ATTR[jkey]
            updates[attr] = _normalize_attr_value(attr, raw)

        merged_patch = {**_PATCH, **updates}
        _PATCH.clear()
        _PATCH.update(merged_patch)
        return _effective_unlocked()


def reset_perf() -> ServiceConfig:
    with _LOCK:
        _PATCH.clear()
        return _effective_unlocked()


def landmark_batch_size_from_env_fallback() -> int:
    import os

    raw = os.environ.get("LANDMARK_BATCH_SIZE", "1").strip() or "1"
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def effective_landmark_batch_size() -> int:
    with _LOCK:
        if _BASELINE is None:
            return landmark_batch_size_from_env_fallback()
        return int(_effective_unlocked().landmark_batch_size)
