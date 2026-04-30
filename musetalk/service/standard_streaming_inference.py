"""
Feature-flagged standard-path streaming: decode frames in memory, landmarks,
UNet/VAE, blend frames directly into FFmpeg rawvideo stdin (no intermediate
result PNG directory or imageio frame list).
"""

from __future__ import annotations

import copy
import glob
import os
import pickle
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import cv2
import numpy as np
import torch
from tqdm import tqdm

from musetalk.service.ffmpeg_pipe import (
    FFmpegRawVideoWriter,
    has_nvenc_encoder,
    has_scale_cuda_filter,
    mux_video_with_audio,
    even_dim,
)
from musetalk.service.resolution_scale import parse_resolution_scale
from musetalk.utils.preprocessing import (
    coord_placeholder,
    get_bbox_range_from_frames,
    get_landmark_and_bbox_from_frames,
    read_imgs,
)
from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import datagen, get_file_type, get_video_fps


def run_standard_streaming_inference(
    *,
    audio_path: str,
    video_path: str,
    bbox_shift: float,
    extra_margin: int,
    parsing_mode: str,
    left_cheek_width: int,
    right_cheek_width: int,
    resolution_scale: str,
    args: Any,
    job_tag: str,
    output_basename: str,
    temp_dir: str,
    result_img_save_path: str,
    crop_coord_save_path: str,
    output_vid_name: str,
    device: torch.device,
    vae: Any,
    unet: Any,
    pe: Any,
    whisper: Any,
    audio_processor: Any,
    weight_dtype: torch.dtype,
    timesteps: torch.Tensor,
    enable_audio_frame_overlap: bool,
    streaming_pipe_buffer_frames: int,
    ffmpeg_video_encoder: str,
    ffmpeg_encoder_preset: str,
    ffmpeg_encoder_crf: str,
    ffmpeg_encoder_cq: str,
    ffmpeg_use_gpu_scale: bool,
    _mark: Callable[..., None],
    _set_stage: Callable[[str], None],
    _finish_stage: Callable[[], None],
    _format_stage_report: Callable[[], str],
) -> tuple[str, str]:
    """Returns (output_vid_name, bbox_shift_text)."""
    if os.path.isdir(result_img_save_path):
        shutil.rmtree(result_img_save_path, ignore_errors=True)
    os.makedirs(result_img_save_path, exist_ok=True)

    scale = parse_resolution_scale(resolution_scale)
    full_target_hw: tuple[int, int] | None = None
    fps = float(args.fps)
    input_img_list: list[str] = []

    if get_file_type(video_path) == "video":
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        frames: list[np.ndarray] = []
        first_full_wh: tuple[int, int] | None = None
        while True:
            ret, fr = cap.read()
            if not ret:
                break
            if first_full_wh is None:
                first_full_wh = (int(fr.shape[1]), int(fr.shape[0]))
            if scale < 1.0 - 1e-9:
                fw, fh = first_full_wh
                nw = even_dim(int(round(fw * scale)))
                nh = even_dim(int(round(fh * scale)))
                fr = cv2.resize(fr, (nw, nh), interpolation=cv2.INTER_AREA)
            frames.append(fr)
        cap.release()
        if not frames:
            raise RuntimeError("video decode produced no frames")
        if scale < 1.0 - 1e-9 and first_full_wh is not None:
            full_target_hw = first_full_wh
        fps = float(get_video_fps(video_path))
        _mark(
            "frame_extract_or_list",
            n_frames=len(frames),
            fps=fps,
            source="video_stream",
        )
    else:
        input_img_list = sorted(
            glob.glob(os.path.join(video_path, "*.[jpJP][pnPN]*[gG]")),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
        )
        frames = read_imgs(input_img_list)
        if not frames:
            raise RuntimeError(f"no images in folder {video_path!r}")
        if scale < 1.0 - 1e-9:
            fh0, fw0 = frames[0].shape[:2]
            full_target_hw = (fw0, fh0)
            nw = even_dim(int(round(fw0 * scale)))
            nh = even_dim(int(round(fh0 * scale)))
            frames = [
                cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA) for f in frames
            ]
        _mark(
            "frame_extract_or_list",
            n_frames=len(frames),
            fps=fps,
            source="image_folder_stream",
        )

    _mark("resolution_downscale", scale=scale, full_target_hw=full_target_hw, streaming=True)

    def _audio_prep() -> list:
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
        return audio_processor.get_whisper_chunk(
            whisper_input_features,
            device,
            weight_dtype,
            whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=args.audio_padding_length_left,
            audio_padding_length_right=args.audio_padding_length_right,
        )

    audio_future = None
    audio_pool = None
    if enable_audio_frame_overlap:
        audio_pool = ThreadPoolExecutor(max_workers=1)
        audio_future = audio_pool.submit(_audio_prep)
        _mark("whisper_audio_submitted", overlap=True)
    else:
        whisper_chunks = _audio_prep()
        _mark("whisper_audio_and_chunks", n_chunks=len(whisper_chunks), overlap=False)

    if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
        print("using extracted coordinates", flush=True)
        used_saved_coord = True
        with open(crop_coord_save_path, "rb") as f:
            coord_list = pickle.load(f)
        if get_file_type(video_path) == "video":
            frame_list = frames
        else:
            frame_list = read_imgs(input_img_list)
    else:
        print("extracting landmarks...time consuming", flush=True)
        used_saved_coord = False
        coord_list, frame_list = get_landmark_and_bbox_from_frames(frames, bbox_shift)
        with open(crop_coord_save_path, "wb") as f:
            pickle.dump(coord_list, f)

    bbox_shift_text = ""#get_bbox_range_from_frames(frame_list, bbox_shift)
    _mark(
        "landmarks_read_or_extract",
        n_coords=len(coord_list),
        n_frames=len(frame_list),
        used_saved_coord=used_saved_coord,
    )

    fp = FaceParsing(
        left_cheek_width=left_cheek_width,
        right_cheek_width=right_cheek_width,
    )
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        y2 = y2 + args.extra_margin
        y2 = min(y2, frame.shape[0])
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)
    _mark("vae_encode_face_crops", n_latents=len(input_latent_list), streaming=True)

    if audio_future is not None:
        try:
            whisper_chunks = audio_future.result()
        finally:
            audio_pool.shutdown(wait=True)
        _mark("whisper_audio_and_chunks", n_chunks=len(whisper_chunks), overlap=True)

    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

    print("start inference (streaming export)", flush=True)
    _set_stage("inference")
    video_num = len(whisper_chunks)
    batch_size = args.batch_size
    gen = datagen(
        whisper_chunks=whisper_chunks,
        vae_encode_latents=input_latent_list_cycle,
        batch_size=batch_size,
        delay_frame=0,
        device=device,
    )
    res_frame_list = []
    for _, (whisper_batch, latent_batch) in enumerate(
        tqdm(gen, total=int(np.ceil(float(video_num) / batch_size)))
    ):
        audio_feature_batch = pe(whisper_batch)
        latent_batch = latent_batch.to(dtype=weight_dtype)
        pred_latents = unet.model(
            latent_batch, timesteps, encoder_hidden_states=audio_feature_batch
        ).sample
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)
    _mark("unet_decode_batches", n_out_frames=len(res_frame_list), streaming=True)

    _set_stage("pad")
    h0, w0 = frame_list_cycle[0].shape[:2]
    src_w, src_h = even_dim(w0), even_dim(h0)
    target_wh = None
    if full_target_hw is not None:
        fw, fh = full_target_hw
        target_wh = (even_dim(fw), even_dim(fh))
    ew, eh = target_wh if target_wh is not None else (src_w, src_h)
    temp_mp4 = os.path.join(temp_dir, f"stream_temp_{job_tag}.mp4")
    buf = max(256, int(streaming_pipe_buffer_frames) * src_w * src_h * 3)
    video_encoder = ffmpeg_video_encoder
    if video_encoder.endswith("_nvenc") and not has_nvenc_encoder():
        print(
            f"[inference] requested encoder '{video_encoder}' unavailable; fallback=libx264",
            flush=True,
        )
        video_encoder = "libx264"
    use_gpu_scale_effective = bool(ffmpeg_use_gpu_scale)
    if target_wh is not None and use_gpu_scale_effective and video_encoder.endswith("_nvenc"):
        if not has_scale_cuda_filter():
            print(
                "[inference] scale_cuda filter unavailable; fallback=CPU scale in fused encode",
                flush=True,
            )
            use_gpu_scale_effective = False

    def _blend_one(i: int, res_frame: np.ndarray) -> np.ndarray | None:
        bbox = coord_list_cycle[i % (len(coord_list_cycle))]
        ori_frame = copy.deepcopy(frame_list_cycle[i % (len(frame_list_cycle))])
        try:
            x1f, y1f, x2f, y2f = [float(v) for v in bbox]
            x1, y1, x2, y2 = map(int, [round(x1f), round(y1f), round(x2f), round(y2f)])
        except Exception as e:
            print(f"[streaming] invalid bbox at frame {i}: {bbox!r} ({e})", flush=True)
            return None
        y2 = y2 + int(args.extra_margin)
        hh, ww = ori_frame.shape[:2]
        x1 = max(0, min(x1, ww - 2))
        x2 = max(x1 + 2, min(x2, ww))
        y1 = max(0, min(y1, hh - 2))
        y2 = max(y1 + 2, min(y2, hh))
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        except Exception as e:
            print(f"[streaming] resize failed frame {i}: {e}", flush=True)
            traceback.print_exc()
            return None
        combine_frame = get_image(
            ori_frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp
        )
        if combine_frame.shape[1] != src_w or combine_frame.shape[0] != src_h:
            combine_frame = cv2.resize(
                combine_frame, (src_w, src_h), interpolation=cv2.INTER_AREA
            )
        return combine_frame

    writer = FFmpegRawVideoWriter(
        temp_mp4,
        src_w,
        src_h,
        fps=fps,
        codec=video_encoder,
        preset=ffmpeg_encoder_preset,
        crf=ffmpeg_encoder_crf,
        cq=ffmpeg_encoder_cq,
        scale_to=target_wh,
        use_gpu_scale=use_gpu_scale_effective,
        bufsize=buf,
    )
    try:
        for i, res_frame in enumerate(tqdm(res_frame_list)):
            fr = _blend_one(i, res_frame)
            if fr is not None:
                writer.write_frame_bgr(fr)
    finally:
        writer.close()

    _mark(
        "ffmpeg_rawvideo_encode",
        path=temp_mp4,
        streaming=True,
        encoder=video_encoder,
        preset=ffmpeg_encoder_preset,
        target=target_wh,
        gpu_scale=use_gpu_scale_effective,
    )

    _set_stage("export")
    if full_target_hw is not None:
        _mark(
            "ffmpeg_upscale_fused_into_encode",
            target=f"{target_wh[0]}x{target_wh[1]}",
            streaming=True,
        )
    mux_video_with_audio(temp_mp4, audio_path, output_vid_name)
    _mark("ffmpeg_mux_final", path=output_vid_name, streaming=True)

    if os.path.exists(temp_mp4):
        os.remove(temp_mp4)

    print(f"result is save to {output_vid_name}", flush=True)
    _finish_stage()
    stage_report = _format_stage_report()
    if stage_report:
        bbox_shift_text = f"{bbox_shift_text}\nStage times: {stage_report}\nmode=streaming_standard"
    _mark("inference_done", out=output_vid_name, streaming=True)
    return output_vid_name, bbox_shift_text
