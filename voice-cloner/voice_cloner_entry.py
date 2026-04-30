"""
Entry point for loading the voice-cloner FastAPI app from another process (monolith mount).

Ensures the voice-cloner directory is on sys.path so `main` resolves to this package's main.py.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from main import app as voice_cloner_app  # noqa: E402

__all__ = ["voice_cloner_app"]
