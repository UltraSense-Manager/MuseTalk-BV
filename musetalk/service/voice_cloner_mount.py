"""Mount in-repo voice-cloner FastAPI app under /api/voice when enabled."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI


def mount_voice_cloner_if_enabled(app: "FastAPI", *, enabled: bool) -> bool:
    """
    Mount voice-cloner at /api/voice. Returns True if mounted.

    Requires JWT_SECRET (same env as MuseTalk) because voice-cloner auth decodes JWTs.
    """
    if not enabled:
        return False
    if not os.environ.get("JWT_SECRET", "").strip():
        print(
            "[voice-cloner] ENABLE_VOICE_CLONER is on but JWT_SECRET is empty; "
            "skipping voice-cloner mount.",
            flush=True,
        )
        return False

    repo_root = Path(__file__).resolve().parents[2]
    entry = repo_root / "voice-cloner" / "voice_cloner_entry.py"
    if not entry.is_file():
        print(f"[voice-cloner] missing entry file: {entry}", flush=True)
        return False

    spec = importlib.util.spec_from_file_location("voice_cloner_entry", entry)
    if spec is None or spec.loader is None:
        print("[voice-cloner] failed to load voice_cloner_entry spec", flush=True)
        return False
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    vc_app = getattr(mod, "voice_cloner_app", None)
    if vc_app is None:
        print("[voice-cloner] voice_cloner_app not exported", flush=True)
        return False

    app.mount("/api/voice", vc_app)
    print("[voice-cloner] mounted at /api/voice", flush=True)
    return True
