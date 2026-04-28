import base64
import hashlib
import logging
import os
import shutil
import tempfile
import wave

logger = logging.getLogger(__name__)


def safe_remove(path: str) -> None:
    """Remove a file; no-op if missing, log on error."""
    try:
        os.remove(path)
        logger.debug("cleanup: removed %s", path)
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("cleanup: could not remove %s: %s", path, e)


def safe_rmtree(path: str) -> None:
    """Remove a directory tree; no-op if missing, log on error."""
    try:
        shutil.rmtree(path)
        logger.debug("cleanup: removed tree %s", path)
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("cleanup: could not remove tree %s: %s", path, e)

# Assumed PCM format for incoming base64 audio (PCM16 mono @ 16kHz by default)
PCM_SAMPLE_RATE = int(os.getenv("PCM_SAMPLE_RATE", "16000"))
PCM_CHANNELS = int(os.getenv("PCM_CHANNELS", "1"))
PCM_SAMPLE_WIDTH = int(os.getenv("PCM_SAMPLE_WIDTH", "2"))  # bytes per sample
MAX_DURATION_SEC = float(os.getenv("MAX_AUDIO_SECONDS", "20"))
MAX_PCM_BYTES = int(PCM_SAMPLE_RATE * PCM_CHANNELS * PCM_SAMPLE_WIDTH * MAX_DURATION_SEC)


def compute_trained_voice_id(reference_b64: str) -> str:
    digest = hashlib.sha256(reference_b64.encode("utf-8")).hexdigest()
    voice_id = digest[:6]
    logger.debug("compute_trained_voice_id: %s", voice_id)
    return voice_id


def append_pcm_chunk(
    existing_b64: str, new_chunk_b64: str
) -> tuple[str, bool]:
    """
    Append a new base64-encoded PCM chunk onto existing PCM, enforcing a 20s cap.

    Returns (combined_b64, is_complete) where is_complete is True if the
    new chunk alone or the combined result reaches/exceeds MAX_DURATION_SEC.
    """
    existing_bytes = base64.b64decode(existing_b64) if existing_b64 else b""
    new_bytes = base64.b64decode(new_chunk_b64)

    total = existing_bytes + new_bytes
    is_complete = False

    if len(new_bytes) >= MAX_PCM_BYTES or len(total) >= MAX_PCM_BYTES:
        total = total[:MAX_PCM_BYTES]
        is_complete = True
        logger.debug("append_pcm_chunk: capped at %s bytes, is_complete=True", MAX_PCM_BYTES)

    combined_b64 = base64.b64encode(total).decode("utf-8")
    logger.debug("append_pcm_chunk: total_bytes=%s is_complete=%s", len(total), is_complete)
    return combined_b64, is_complete


def write_b64_audio_to_temp_wav(prefix: str, audio_b64: str) -> str:
    """
    Decode base64-encoded PCM audio and write it to a temporary WAV file,
    truncated to MAX_DURATION_SEC if needed.
    """
    pcm_bytes = base64.b64decode(audio_b64)
    original_len = len(pcm_bytes)

    if len(pcm_bytes) > MAX_PCM_BYTES:
        pcm_bytes = pcm_bytes[:MAX_PCM_BYTES]
        logger.debug("write_b64_audio_to_temp_wav: truncated %s -> %s bytes", original_len, len(pcm_bytes))

    fd, path = tempfile.mkstemp(prefix=f"{prefix}_", suffix=".wav")
    os.close(fd)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(PCM_CHANNELS)
        wf.setsampwidth(PCM_SAMPLE_WIDTH)
        wf.setframerate(PCM_SAMPLE_RATE)
        wf.writeframes(pcm_bytes)

    logger.debug("write_b64_audio_to_temp_wav: wrote %s bytes to %s", len(pcm_bytes), path)
    return path

