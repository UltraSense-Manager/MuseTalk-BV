import os
import time
import pdb
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import gradio as gr
import numpy as np
import sys
import subprocess

from huggingface_hub import snapshot_download
import requests

import argparse
import os
import traceback
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
from tqdm import tqdm
import copy
from argparse import Namespace
import shutil
import gdown
import imageio
import ffmpeg
from moviepy.editor import *
from transformers import WhisperModel

ProjectDir = os.path.abspath(os.path.dirname(__file__))
CheckpointsDir = os.path.join(ProjectDir, "models")

@torch.no_grad()
def debug_inpainting(video_path, bbox_shift, extra_margin=10, parsing_mode="jaw", 
                    left_cheek_width=90, right_cheek_width=90):
    """Debug inpainting parameters, only process the first frame"""
    # Set default parameters
    args_dict = {
        "result_dir": './results/debug', 
        "fps": 25, 
        "batch_size": 1, 
        "output_vid_name": '', 
        "use_saved_coord": False,
        "audio_padding_length_left": 2,
        "audio_padding_length_right": 2,
        "version": "v15",
        "extra_margin": extra_margin,
        "parsing_mode": parsing_mode,
        "left_cheek_width": left_cheek_width,
        "right_cheek_width": right_cheek_width
    }
    args = Namespace(**args_dict)

    # Create debug directory
    os.makedirs(args.result_dir, exist_ok=True)
    
    # Read first frame
    if get_file_type(video_path) == "video":
        reader = imageio.get_reader(video_path)
        first_frame = reader.get_data(0)
        reader.close()
    else:
        first_frame = cv2.imread(video_path)
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    # Save first frame
    debug_frame_path = os.path.join(args.result_dir, "debug_frame.png")
    cv2.imwrite(debug_frame_path, cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
    
    # Get face coordinates
    coord_list, frame_list = get_landmark_and_bbox([debug_frame_path], bbox_shift)
    bbox = coord_list[0]
    frame = frame_list[0]
    
    if bbox == coord_placeholder:
        return None, "No face detected, please adjust bbox_shift parameter"
    
    # Initialize face parser
    fp = FaceParsing(
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width
    )
    
    # Process first frame
    x1, y1, x2, y2 = bbox
    y2 = y2 + args.extra_margin
    y2 = min(y2, frame.shape[0])
    crop_frame = frame[y1:y2, x1:x2]
    crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
    
    # Generate random audio features
    random_audio = torch.randn(1, 50, 384, device=device, dtype=weight_dtype)
    audio_feature = pe(random_audio)
    
    # Get latents
    latents = vae.get_latents_for_unet(crop_frame)
    latents = latents.to(dtype=weight_dtype)
    
    # Generate prediction results
    pred_latents = unet.model(latents, timesteps, encoder_hidden_states=audio_feature).sample
    recon = vae.decode_latents(pred_latents)
    
    # Inpaint back to original image
    res_frame = recon[0]
    res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
    combine_frame = get_image(frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
    
    # Save results (no need to convert color space again since get_image already returns RGB format)
    debug_result_path = os.path.join(args.result_dir, "debug_result.png")
    cv2.imwrite(debug_result_path, combine_frame)
    
    # Create information text
    info_text = f"Parameter information:\n" + \
                f"bbox_shift: {bbox_shift}\n" + \
                f"extra_margin: {extra_margin}\n" + \
                f"parsing_mode: {parsing_mode}\n" + \
                f"left_cheek_width: {left_cheek_width}\n" + \
                f"right_cheek_width: {right_cheek_width}\n" + \
                f"Detected face coordinates: [{x1}, {y1}, {x2}, {y2}]"
    
    return cv2.cvtColor(combine_frame, cv2.COLOR_RGB2BGR), info_text

def print_directory_contents(path):
    for child in os.listdir(path):
        child_path = os.path.join(path, child)
        if os.path.isdir(child_path):
            print(child_path)

def download_model():
    # 检查必需的模型文件是否存在
    required_models = {
        "MuseTalk": f"{CheckpointsDir}/musetalkV15/unet.pth",
        "MuseTalk": f"{CheckpointsDir}/musetalkV15/musetalk.json",
        "SD VAE": f"{CheckpointsDir}/sd-vae/config.json",
        "Whisper": f"{CheckpointsDir}/whisper/config.json",
        "DWPose": f"{CheckpointsDir}/dwpose/dw-ll_ucoco_384.pth",
        "SyncNet": f"{CheckpointsDir}/syncnet/latentsync_syncnet.pt",
        "Face Parse": f"{CheckpointsDir}/face-parse-bisent/79999_iter.pth",
        "ResNet": f"{CheckpointsDir}/face-parse-bisent/resnet18-5c106cde.pth"
    }
    
    missing_models = []
    for model_name, model_path in required_models.items():
        if not os.path.exists(model_path):
            missing_models.append(model_name)
    
    if missing_models:
        # 全用英文
        print("The following required model files are missing:")
        for model in missing_models:
            print(f"- {model}")
        print("\nPlease run the download script to download the missing models:")
        if sys.platform == "win32":
            print("Windows: Run download_weights.bat")
        else:
            print("Linux/Mac: Run ./download_weights.sh")
        sys.exit(1)
    else:
        print("All required model files exist.")




download_model()  # for huggingface deployment.

from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import (
    get_landmark_and_bbox,
    get_landmark_and_bbox_with_range_from_frames,
    read_imgs,
    coord_placeholder,
    get_bbox_range,
)
from musetalk.service.config import load_service_config
from musetalk.service.resolution_scale import (
    downscale_png_dir_inplace,
    parse_resolution_scale,
    upscale_video_stream,
)


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


@torch.no_grad()
def inference(
    audio_path,
    video_path,
    bbox_shift,
    extra_margin=10,
    parsing_mode="jaw",
    left_cheek_width=90,
    right_cheek_width=90,
    resolution_scale: str = "full",
    status_callback: Callable[[str, float, dict[str, float]], None] | None = None,
    progress=gr.Progress(track_tqdm=True),
):
    print(
        f"[inference] start audio={audio_path} video={video_path} "
        f"bbox_shift={bbox_shift}",
        flush=True,
    )
    _t0 = time.perf_counter()
    _tp = [_t0]
    stage_times: dict[str, float] = {}
    _stage_name = [""]
    _stage_t0 = [time.perf_counter()]

    def _mark(phase: str, **kv: object) -> None:
        now = time.perf_counter()
        extra = (" " + " ".join(f"{k}={v}" for k, v in kv.items())) if kv else ""
        print(
            f"[inference:timing] {phase} dt={now - _tp[0]:.3f}s total={now - _t0:.3f}s"
            f"{extra}",
            flush=True,
        )
        _tp[0] = now

    def _set_stage(stage: str) -> None:
        now = time.perf_counter()
        prev = _stage_name[0]
        if prev:
            stage_times[prev] = stage_times.get(prev, 0.0) + (now - _stage_t0[0])
            if status_callback:
                status_callback(prev, stage_times[prev], dict(stage_times))
        _stage_name[0] = stage
        _stage_t0[0] = now
        if status_callback:
            status_callback(stage, stage_times.get(stage, 0.0), dict(stage_times))

    def _finish_stage() -> None:
        now = time.perf_counter()
        prev = _stage_name[0]
        if prev:
            stage_times[prev] = stage_times.get(prev, 0.0) + (now - _stage_t0[0])
            if status_callback:
                status_callback(prev, stage_times[prev], dict(stage_times))

    def _format_stage_report() -> str:
        order = ("upload", "preprocess", "inference", "pad", "export")
        parts = []
        for name in order:
            if name in stage_times:
                parts.append(f"{name}={stage_times[name]:.2f}s")
        for name, val in stage_times.items():
            if name not in order:
                parts.append(f"{name}={val:.2f}s")
        return " | ".join(parts)

    # Set default parameters, aligned with inference.py
    args_dict = {
        "result_dir": './results/output', 
        "fps": 25, 
        "batch_size": int(getattr(globals().get("svc_cfg"), "standard_batch_size", 8)),
        "output_vid_name": '', 
        "use_saved_coord": False,
        "audio_padding_length_left": 2,
        "audio_padding_length_right": 2,
        "version": "v15",  # Fixed use v15 version
        "extra_margin": extra_margin,
        "parsing_mode": parsing_mode,
        "left_cheek_width": left_cheek_width,
        "right_cheek_width": right_cheek_width
    }
    args = Namespace(**args_dict)
    cfg = globals().get("svc_cfg")
    cpu_workers = max(1, int(getattr(cfg, "cpu_workers", 2)))
    enable_parallel_blend = bool(getattr(cfg, "enable_parallel_blend", False))
    enable_audio_frame_overlap = bool(
        getattr(cfg, "enable_parallel_audio_frame_overlap", True)
    )
    _set_stage("preprocess")

    # Check ffmpeg
    if not fast_check_ffmpeg():
        print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed", flush=True)

    input_basename = os.path.basename(video_path).split('.')[0]
    audio_basename = os.path.basename(audio_path).split('.')[0]
    # API uploads use stable names like input_video/input_audio. Add a deterministic
    # per-job suffix from full paths to avoid collisions across concurrent/sequential jobs.
    job_tag = hashlib.sha256(f"{video_path}|{audio_path}".encode("utf-8")).hexdigest()[:12]
    output_basename = f"{input_basename}_{audio_basename}_{job_tag}"
    
    # Create temporary directory
    temp_dir = os.path.join(args.result_dir, f"{args.version}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Set result save path
    result_img_save_path = os.path.join(temp_dir, output_basename)
    crop_coord_save_path = os.path.join(args.result_dir, "../", f"{input_basename}_{job_tag}.pkl")
    if os.path.isdir(result_img_save_path):
        shutil.rmtree(result_img_save_path, ignore_errors=True)
    os.makedirs(result_img_save_path, exist_ok=True)

    if args.output_vid_name == "":
        output_vid_name = os.path.join(temp_dir, output_basename+".mp4")
    else:
        output_vid_name = os.path.join(temp_dir, args.output_vid_name)

    use_std_stream = bool(
        getattr(cfg, "enable_streaming_standard", False)
    ) and fast_check_ffmpeg()
    if use_std_stream:
        print("[inference] pipeline=streaming_standard", flush=True)
        from musetalk.service.standard_streaming_inference import (
            run_standard_streaming_inference,
        )

        return run_standard_streaming_inference(
            audio_path=audio_path,
            video_path=video_path,
            bbox_shift=bbox_shift,
            extra_margin=extra_margin,
            parsing_mode=parsing_mode,
            left_cheek_width=left_cheek_width,
            right_cheek_width=right_cheek_width,
            resolution_scale=resolution_scale,
            args=args,
            job_tag=job_tag,
            output_basename=output_basename,
            temp_dir=temp_dir,
            result_img_save_path=result_img_save_path,
            crop_coord_save_path=crop_coord_save_path,
            output_vid_name=output_vid_name,
            device=device,
            vae=vae,
            unet=unet,
            pe=pe,
            whisper=whisper,
            audio_processor=audio_processor,
            weight_dtype=weight_dtype,
            timesteps=timesteps,
            enable_audio_frame_overlap=enable_audio_frame_overlap,
            streaming_pipe_buffer_frames=int(
                getattr(cfg, "streaming_pipe_buffer_frames", 4)
            ),
            _mark=_mark,
            _set_stage=_set_stage,
            _finish_stage=_finish_stage,
            _format_stage_report=_format_stage_report,
        )

    ############################################## extract frames from source video ##############################################
    if get_file_type(video_path) == "video":
        save_dir_full = os.path.join(temp_dir, f"{input_basename}_{job_tag}")
        if os.path.isdir(save_dir_full):
            shutil.rmtree(save_dir_full, ignore_errors=True)
        os.makedirs(save_dir_full, exist_ok=True)
        # Read video
        reader = imageio.get_reader(video_path)

        # Save images
        for i, im in enumerate(reader):
            imageio.imwrite(f"{save_dir_full}/{i:08d}.png", im)
        input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
        fps = get_video_fps(video_path)
    else: # input img folder
        input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        fps = args.fps

    _mark(
        "frame_extract_or_list",
        n_frames=len(input_img_list),
        fps=fps,
        source="video" if get_file_type(video_path) == "video" else "image_folder",
    )

    try:
        scale = parse_resolution_scale(resolution_scale)
    except ValueError as e:
        raise ValueError(str(e)) from e
    full_target_hw: tuple[int, int] | None = None
    if scale < 1.0 - 1e-9:
        if get_file_type(video_path) == "video":
            full_target_hw = downscale_png_dir_inplace(save_dir_full, scale)
        else:
            full_target_hw = downscale_png_dir_inplace(str(video_path), scale)
        print(
            f"[inference] resolution_scale={resolution_scale!r} "
            f"(scale={scale}) upscale target={full_target_hw}",
            flush=True,
        )
    _mark(
        "resolution_downscale",
        scale=scale,
        full_target_hw=full_target_hw,
    )

    ############################################## extract audio feature ##############################################
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

    ############################################## preprocess input image  ##############################################
    if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
        print("using extracted coordinates", flush=True)
        used_saved_coord = True
        with open(crop_coord_save_path,'rb') as f:
            coord_list = pickle.load(f)
        frame_list = read_imgs(input_img_list)
    else:
        print("extracting landmarks...time consuming", flush=True)
        used_saved_coord = False
        frame_list = read_imgs(input_img_list)
        coord_list, frame_list, bbox_shift_text = get_landmark_and_bbox_with_range_from_frames(
            frame_list, bbox_shift
        )
        with open(crop_coord_save_path, 'wb') as f:
            pickle.dump(coord_list, f)
    if used_saved_coord:
        bbox_shift_text = get_bbox_range(input_img_list, bbox_shift)
    _mark(
        "landmarks_read_or_extract",
        n_coords=len(coord_list),
        n_frames=len(frame_list),
        used_saved_coord=used_saved_coord,
    )

    # Initialize face parser
    fp = FaceParsing(
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width
    )
    
    i = 0
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        y2 = y2 + args.extra_margin
        y2 = min(y2, frame.shape[0])
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)
    _mark("vae_encode_face_crops", n_latents=len(input_latent_list))

    if audio_future is not None:
        try:
            whisper_chunks = audio_future.result()
        finally:
            audio_pool.shutdown(wait=True)
        _mark("whisper_audio_and_chunks", n_chunks=len(whisper_chunks), overlap=True)

    # to smooth the first and the last frame
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    
    ############################################## inference batch by batch ##############################################
    print("start inference", flush=True)
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
    for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
        audio_feature_batch = pe(whisper_batch)
        # Ensure latent_batch is consistent with model weight type
        latent_batch = latent_batch.to(dtype=weight_dtype)
        
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)
    _mark("unet_decode_batches", n_out_frames=len(res_frame_list))

    ############################################## pad to full image ##############################################
    print("pad talking image to original video", flush=True)
    _set_stage("pad")

    def _blend_and_write(i: int, res_frame) -> bool:
        bbox = coord_list_cycle[i%(len(coord_list_cycle))]
        ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
        try:
            x1f, y1f, x2f, y2f = [float(v) for v in bbox]
            x1, y1, x2, y2 = map(int, [round(x1f), round(y1f), round(x2f), round(y2f)])
        except Exception as e:
            print(f"[inference] invalid bbox at frame {i}: {bbox!r} ({e})", flush=True)
            return False

        y2 = y2 + int(args.extra_margin)
        # Clamp bbox into current frame bounds and ensure at least 2x2 crop.
        h, w = ori_frame.shape[:2]
        x1 = max(0, min(x1, w - 2))
        x2 = max(x1 + 2, min(x2, w))
        y1 = max(0, min(y1, h - 2))
        y2 = max(y1 + 2, min(y2, h))
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
        except Exception as e:
            print(f"[inference] resize failed frame {i}: {e}", flush=True)
            traceback.print_exc()
            return False
        
        # Use v15 version blending
        combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
            
        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)
        return True

    written_count = 0
    if enable_parallel_blend and cpu_workers > 1:
        with ThreadPoolExecutor(max_workers=cpu_workers) as pool:
            futures = [pool.submit(_blend_and_write, i, rf) for i, rf in enumerate(res_frame_list)]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                if fut.result():
                    written_count += 1
        _mark(
            "pad_blend_write_png",
            n_expected=len(res_frame_list),
            n_written=written_count,
            parallel=True,
            workers=cpu_workers,
        )
    else:
        for i, res_frame in enumerate(tqdm(res_frame_list)):
            if _blend_and_write(i, res_frame):
                written_count += 1
        _mark(
            "pad_blend_write_png",
            n_expected=len(res_frame_list),
            n_written=written_count,
            parallel=False,
            workers=1,
        )

    # Frame rate
    fps = 25
    _set_stage("export")
    # Output video path
    output_video = os.path.join(temp_dir, f"temp_{job_tag}.mp4")

    # Read images
    def is_valid_image(file):
        pattern = re.compile(r'\d{8}\.png')
        return pattern.match(file)

    images = []
    files = [file for file in os.listdir(result_img_save_path) if is_valid_image(file)]
    files.sort(key=lambda x: int(x.split('.')[0]))

    for file in files:
        filename = os.path.join(result_img_save_path, file)
        images.append(imageio.imread(filename))
    
    if not images:
        raise RuntimeError(
            f"No output frames were generated in {result_img_save_path}; cannot encode video."
        )

    # imageio/ffmpeg requires all frames to have identical HxW.
    target_h, target_w = images[0].shape[:2]
    normalized_images = []
    for idx, img in enumerate(images):
        if img.shape[:2] != (target_h, target_w):
            print(
                f"warning: frame size mismatch at index {idx}: "
                f"{img.shape[:2]} -> {(target_h, target_w)}"
            , flush=True)
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        normalized_images.append(img)
    _mark("load_pngs_normalize", n_images=len(normalized_images))

    if full_target_hw is not None:
        fw, fh = full_target_hw
        print(
            f"[inference] will upscale encoded temp video via ffmpeg to {fw}x{fh}",
            flush=True,
        )
    _mark(
        "upscale_frames_optional",
        did_upscale=full_target_hw is not None,
        target=full_target_hw,
    )

    # Save video
    imageio.mimwrite(output_video, normalized_images, 'FFMPEG', fps=fps, codec='libx264', pixelformat='yuv420p')
    _mark("imageio_mimwrite_temp_mp4", path=output_video)
    if full_target_hw is not None:
        fw, fh = full_target_hw
        upscaled_temp = os.path.join(temp_dir, f"temp_upscaled_{job_tag}.mp4")
        upscale_video_stream(output_video, fw, fh, upscaled_temp)
        os.replace(upscaled_temp, output_video)
        _mark("ffmpeg_upscale_video_stream", target=f"{fw}x{fh}")
    else:
        _mark("ffmpeg_upscale_video_stream_skipped")

    input_video = output_video
    # Check if the input_video and audio_path exist
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video file not found: {input_video}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Read video
    reader = imageio.get_reader(input_video)
    fps = reader.get_meta_data()['fps']  # Get original video frame rate
    reader.close() # Otherwise, error on win11: PermissionError: [WinError 32] Another program is using this file, process cannot access. : 'temp.mp4'
    # Store frames in list
    frames = normalized_images
    
    print(f"[inference] total frames={len(frames)}", flush=True)

    # Load the video
    video_clip = VideoFileClip(input_video)

    # Load the audio
    audio_clip = AudioFileClip(audio_path)

    # Set the audio to the video
    video_clip = video_clip.set_audio(audio_clip)

    # Write the output video
    video_clip.write_videofile(output_vid_name, codec='libx264', audio_codec='aac',fps=25)
    video_clip.close()
    audio_clip.close()
    _mark("moviepy_write_final_mp4", path=output_vid_name)

    if os.path.exists(output_video):
        os.remove(output_video)
    #shutil.rmtree(result_img_save_path)
    print(f"result is save to {output_vid_name}", flush=True)
    _finish_stage()
    stage_report = _format_stage_report()
    if stage_report:
        bbox_shift_text = f"{bbox_shift_text}\nStage times: {stage_report}"
    _mark("inference_done", out=output_vid_name)
    return output_vid_name, bbox_shift_text



# load model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae, unet, pe = load_all_model(
    unet_model_path="./models/musetalkV15/unet.pth", 
    vae_type="sd-vae",
    unet_config="./models/musetalkV15/musetalk.json",
    device=device
)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ffmpeg_path", type=str, default=r"ffmpeg-master-latest-win64-gpl-shared\bin", help="Path to ffmpeg executable")
parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
parser.add_argument("--share", action="store_true", help="Create a public link")
parser.add_argument("--use_float16", action="store_true", help="Use float16 for faster inference")
args = parser.parse_args()

# Set data type
if args.use_float16:
    # Convert models to half precision for better performance
    pe = pe.half()
    vae.vae = vae.vae.half()
    unet.model = unet.model.half()
    weight_dtype = torch.float16
else:
    weight_dtype = torch.float32

# Move models to specified device
pe = pe.to(device)
vae.vae = vae.vae.to(device)
unet.model = unet.model.to(device)

timesteps = torch.tensor([0], device=device)

# Initialize audio processor and Whisper model
audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
whisper = WhisperModel.from_pretrained("./models/whisper")
whisper = whisper.to(device=device, dtype=weight_dtype).eval()
whisper.requires_grad_(False)


def _realtime_api_runner(
    work_dir: str,
    video_path: str,
    audio_path: str,
    job_id: str,
    persist_avatar_id: str,
    reuse_avatar: bool,
    bbox_shift: float,
    extra_margin: int,
    parsing_mode: str,
    left_cheek_width: int,
    right_cheek_width: int,
    prep_frames: int,
    batch_size: int,
    fps: int,
    resolution_scale: str = "full",
    status_callback: Callable[[str, float, dict[str, float]], None] | None = None,
) -> tuple[str, str, str]:
    from musetalk.service.realtime_job import RealtimeJobContext, run_realtime_job

    ctx = RealtimeJobContext(
        version="v15",
        extra_margin=extra_margin,
        parsing_mode=parsing_mode,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
        skip_save_images=False,
        left_cheek_width=left_cheek_width,
        right_cheek_width=right_cheek_width,
        vae=vae,
        unet=unet,
        pe=pe,
        whisper=whisper,
        audio_processor=audio_processor,
        fp=FaceParsing(
            left_cheek_width=left_cheek_width,
            right_cheek_width=right_cheek_width,
        ),
        device=device,
        weight_dtype=weight_dtype,
        timesteps=timesteps,
        cpu_workers=max(1, int(getattr(svc_cfg, "cpu_workers", 2))),
        enable_parallel_realtime_prep=bool(
            getattr(svc_cfg, "enable_parallel_realtime_prep", False)
        ),
        enable_streaming_realtime=bool(
            getattr(svc_cfg, "enable_streaming_realtime", False)
        ),
        streaming_pipe_buffer_frames=int(
            getattr(svc_cfg, "streaming_pipe_buffer_frames", 4)
        ),
    )
    return run_realtime_job(
        ctx,
        work_dir,
        video_path,
        audio_path,
        job_id,
        persist_avatar_id,
        reuse_avatar,
        bbox_shift,
        extra_margin,
        parsing_mode,
        left_cheek_width,
        right_cheek_width,
        prep_frames,
        batch_size,
        fps,
        resolution_scale,
        status_callback=status_callback,
    )


def check_video(video):
    if not isinstance(video, str):
        return video # in case of none type
    # Define the output video file name
    dir_path, file_name = os.path.split(video)
    if file_name.startswith("outputxxx_"):
        return video
    # Add the output prefix to the file name
    output_file_name = "outputxxx_" + file_name

    os.makedirs('./results',exist_ok=True)
    os.makedirs('./results/output',exist_ok=True)
    os.makedirs('./results/input',exist_ok=True)

    # Combine the directory path and the new file name
    output_video = os.path.join('./results/input', output_file_name)


    # read video
    reader = imageio.get_reader(video)
    fps = reader.get_meta_data()['fps']  # get fps from original video

    # conver fps to 25
    frames = [im for im in reader]
    target_fps = 25
    
    L = len(frames)
    L_target = int(L / fps * target_fps)
    original_t = [x / fps for x in range(1, L+1)]
    t_idx = 0
    target_frames = []
    for target_t in range(1, L_target+1):
        while target_t / target_fps > original_t[t_idx]:
            t_idx += 1      # find the first t_idx so that target_t / target_fps <= original_t[t_idx]
            if t_idx >= L:
                break
        target_frames.append(frames[t_idx])

    # save video
    imageio.mimwrite(output_video, target_frames, 'FFMPEG', fps=25, codec='libx264', quality=9, pixelformat='yuv420p')
    return output_video




css = """#input_img {max-width: 1024px !important} #output_vid {max-width: 1024px; max-height: 576px}"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(
        """<div align='center'> <h1>MuseTalk: Real-Time High-Fidelity Video Dubbing via Spatio-Temporal Sampling</h1> \
                    <h2 style='font-weight: 450; font-size: 1rem; margin: 0rem'>\
                    </br>\
                    Yue Zhang <sup>*</sup>,\
                    Zhizhou Zhong <sup>*</sup>,\
                    Minhao Liu<sup>*</sup>,\
                    Zhaokang Chen,\
                    Bin Wu<sup>†</sup>,\
                    Yubin Zeng,\
                    Chao Zhang,\
                    Yingjie He,\
                    Junxin Huang,\
                    Wenjiang Zhou <br>\
                    (<sup>*</sup>Equal Contribution, <sup>†</sup>Corresponding Author, benbinwu@tencent.com)\
                    Lyra Lab, Tencent Music Entertainment\
                </h2> \
                <a style='font-size:18px;color: #000000' href='https://github.com/TMElyralab/MuseTalk'>[Github Repo]</a>\
                <a style='font-size:18px;color: #000000' href='https://github.com/TMElyralab/MuseTalk'>[Huggingface]</a>\
                <a style='font-size:18px;color: #000000' href='https://arxiv.org/abs/2410.10122'> [Technical report] </a>"""
    )

    with gr.Row():
        with gr.Column():
            audio = gr.Audio(label="Drving Audio",type="filepath")
            video = gr.Video(label="Reference Video",sources=['upload'])
            bbox_shift = gr.Number(label="BBox_shift value, px", value=0)
            extra_margin = gr.Slider(label="Extra Margin", minimum=0, maximum=40, value=10, step=1)
            parsing_mode = gr.Radio(label="Parsing Mode", choices=["jaw", "raw"], value="jaw")
            left_cheek_width = gr.Slider(label="Left Cheek Width", minimum=20, maximum=160, value=90, step=5)
            right_cheek_width = gr.Slider(label="Right Cheek Width", minimum=20, maximum=160, value=90, step=5)
            bbox_shift_scale = gr.Textbox(label="'left_cheek_width' and 'right_cheek_width' parameters determine the range of left and right cheeks editing when parsing model is 'jaw'. The 'extra_margin' parameter determines the movement range of the jaw. Users can freely adjust these three parameters to obtain better inpainting results.")

            with gr.Row():
                debug_btn = gr.Button("1. Test Inpainting ")
                btn = gr.Button("2. Generate")
        with gr.Column():
            debug_image = gr.Image(label="Test Inpainting Result (First Frame)")
            debug_info = gr.Textbox(label="Parameter Information", lines=5)
            out1 = gr.Video()
    
    video.change(
        fn=check_video, inputs=[video], outputs=[video]
    )
    btn.click(
        fn=inference,
        inputs=[
            audio,
            video,
            bbox_shift,
            extra_margin,
            parsing_mode,
            left_cheek_width,
            right_cheek_width
        ],
        outputs=[out1,bbox_shift_scale]
    )
    debug_btn.click(
        fn=debug_inpainting,
        inputs=[
            video,
            bbox_shift,
            extra_margin,
            parsing_mode,
            left_cheek_width,
            right_cheek_width
        ],
        outputs=[debug_image, debug_info]
    )

# Check ffmpeg and add to PATH
if not fast_check_ffmpeg():
    print(f"Adding ffmpeg to PATH: {args.ffmpeg_path}")
    # According to operating system, choose path separator
    path_separator = ';' if sys.platform == 'win32' else ':'
    os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
    if not fast_check_ffmpeg():
        print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

# Solve asynchronous IO issues on Windows
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from musetalk.service.api import create_service_app
from starlette.middleware.cors import CORSMiddleware
import uvicorn

demo.queue()
svc_cfg = load_service_config()
if svc_cfg.secured_mode and (
    not svc_cfg.gradio_username or not svc_cfg.gradio_password
):
    raise RuntimeError(
        "SECURED_MODE is on but Gradio credentials are missing. "
        "Set GRADIO_USER and GRADIO_PASS (recommended), or USER and PASS."
    )

service_app = create_service_app(
    inference, svc_cfg, realtime_runner=_realtime_api_runner
)
if svc_cfg.secured_mode:
    root_app = gr.mount_gradio_app(
        service_app,
        demo,
        path="",
        auth=(svc_cfg.gradio_username, svc_cfg.gradio_password),
    )
else:
    root_app = gr.mount_gradio_app(service_app, demo, path="")

root_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if args.share:
    print(
        "Note: --share is not wired for the FastAPI+Gradio mount; use a tunnel (e.g. ngrok) "
        "or expose --ip 0.0.0.0 behind a reverse proxy."
    )

uvicorn.run(
    root_app,
    host=args.ip,
    port=int(args.port),
    log_level="info",
)
