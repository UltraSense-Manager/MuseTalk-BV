import base64
import logging
import os
import wave

import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

from audio import safe_rmtree

logger = logging.getLogger(__name__)

ckpt_converter = 'checkpoints_v2/converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs_v2'

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)
logger.info("cloner loaded device=%s output_dir=%s", device, output_dir)


def extract_speaker_embedding(audio_path):
    """
    Convenience wrapper to extract speaker embedding and audio name
    for a given audio file path.
    """
    logger.debug("extract_speaker_embedding: %s", audio_path)
    result = se_extractor.get_se(audio_path, tone_color_converter, vad=True)
    logger.debug("extract_speaker_embedding done: %s", audio_path)
    return result


def clone(reference_speaker, base_speaker, target_se=None, target_audio_name=None):
    logger.info("clone: reference_speaker=%s base_speaker=%s", reference_speaker, base_speaker)
    # Always extract source speaker embedding from base audio
    source_se, base_audio_name = se_extractor.get_se(
        base_speaker, tone_color_converter, vad=True
    )
    logger.debug("clone: source_se extracted base_audio_name=%s", base_audio_name)

    # Use precomputed target embedding if available; otherwise compute it now
    if target_se is None or target_audio_name is None:
        target_se, target_audio_name = se_extractor.get_se(
            reference_speaker, tone_color_converter, vad=True
        )
        logger.debug("clone: target_se computed target_audio_name=%s", target_audio_name)

    audio_name = target_audio_name or base_audio_name
    save_path = f'{output_dir}/{audio_name}.wav'
    encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=base_speaker,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path,
        message=encode_message,
    )
    logger.debug("clone: convert wrote %s", save_path)

    # Read the generated WAV and return base64-encoded PCM data
    try:
        with wave.open(save_path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
        pcm_b64 = base64.b64encode(frames).decode("utf-8")
        logger.debug("clone: read %s bytes PCM from %s", len(frames), save_path)
        logger.info("clone: done")
        return pcm_b64
    finally:
        try:
            os.remove(save_path)
            logger.debug("clone: removed temp file %s", save_path)
        except OSError as e:
            logger.warning("clone: could not remove temp file %s: %s", save_path, e)
        safe_rmtree(os.path.join("processed", base_audio_name))