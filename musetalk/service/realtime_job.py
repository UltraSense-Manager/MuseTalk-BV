"""
Realtime MuseTalk pipeline for API: first N frames from video (ffmpeg) for
avatar preparation, then realtime-style batched inference + mux.
"""

from __future__ import annotations

import copy
import glob
import json
import os
import pickle
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.blending import get_image_blending, get_image_prepare_material
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.utils import datagen
from musetalk.service.resolution_scale import (
    downscale_png_dir_inplace,
    parse_resolution_scale,
    upscale_video_replace_audio,
)


def extract_first_frames_ffmpeg(video_path: str, out_dir: str, n: int) -> None:
    """Write first n frames as 00000000.png … using ffmpeg."""
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "%08d.png")
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vframes", str(n), pattern],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg frame extract failed ({proc.returncode}): {proc.stderr[-1200:]!r}"
        )
    files = sorted(glob.glob(os.path.join(out_dir, "*.png")))
    if not files:
        raise RuntimeError("ffmpeg produced no PNG frames")


def _osmakedirs(paths: list[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _video2imgs(vid_path: str, save_path: str, ext: str = ".png", cut_frame: int = 10000000) -> None:
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}{ext}", frame)
            count += 1
        else:
            break
    cap.release()


@dataclass
class RealtimeJobContext:
    version: str
    extra_margin: int
    parsing_mode: str
    audio_padding_length_left: int
    audio_padding_length_right: int
    skip_save_images: bool
    left_cheek_width: int
    right_cheek_width: int
    vae: Any
    unet: Any
    pe: Any
    whisper: Any
    audio_processor: AudioProcessor
    fp: FaceParsing
    device: torch.device
    weight_dtype: torch.dtype
    timesteps: torch.Tensor


class RealtimeAvatar:
    """Non-interactive avatar prep + realtime inference (from scripts/realtime_inference)."""

    @torch.no_grad()
    def __init__(
        self,
        ctx: RealtimeJobContext,
        avatar_id: str,
        video_path: str,
        bbox_shift: float,
        batch_size: int,
        avatar_root: str,
        upscale_target_wh: tuple[int, int] | None = None,
    ) -> None:
        self.ctx = ctx
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.upscale_target_wh = upscale_target_wh
        self.base_path = avatar_root
        self.avatar_path = self.base_path
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info: dict[str, Any] = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift,
            "version": ctx.version,
        }
        if upscale_target_wh is not None:
            uw, uh = upscale_target_wh
            self.avatar_info["upscale_width"] = int(uw)
            self.avatar_info["upscale_height"] = int(uh)
        self.batch_size = batch_size
        self.idx = 0
        if os.path.exists(self.avatar_path):
            shutil.rmtree(self.avatar_path)
        _osmakedirs(
            [self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path]
        )
        self.prepare_material()

    def prepare_material(self) -> None:
        print("preparing realtime avatar materials ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

        if os.path.isfile(self.video_path):
            _video2imgs(self.video_path, self.full_imgs_path, ext="png")
        else:
            files = os.listdir(self.video_path)
            files.sort()
            files = [f for f in files if f.split(".")[-1].lower() == "png"]
            for filename in files:
                shutil.copyfile(
                    os.path.join(self.video_path, filename),
                    os.path.join(self.full_imgs_path, filename),
                )
        input_img_list = sorted(
            glob.glob(os.path.join(self.full_imgs_path, "*.[jpJP][pnPN]*[gG]"))
        )

        print("extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        input_latent_list: list[Any] = []
        idx = -1
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        for bbox, frame in zip(coord_list, frame_list):
            idx += 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            if self.ctx.version == "v15":
                y2 = y2 + self.ctx.extra_margin
                y2 = min(y2, frame.shape[0])
                coord_list[idx] = [x1, y1, x2, y2]
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(
                crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4
            )
            latents = self.ctx.vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        for i, frame in enumerate(tqdm(self.frame_list_cycle, desc="masks")):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)

            x1, y1, x2, y2 = self.coord_list_cycle[i]
            mode = self.ctx.parsing_mode if self.ctx.version == "v15" else "raw"
            mask, crop_box = get_image_prepare_material(
                frame, [x1, y1, x2, y2], fp=self.ctx.fp, mode=mode
            )

            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle += [crop_box]
            self.mask_list_cycle.append(mask)

        with open(self.mask_coords_path, "wb") as f:
            pickle.dump(self.mask_coords_list_cycle, f)

        with open(self.coords_path, "wb") as f:
            pickle.dump(self.coord_list_cycle, f)

        torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path))

        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

    def process_frames(
        self, res_frame_queue: queue.Queue, video_len: int, skip_save_images: bool
    ) -> None:
        while True:
            if self.idx >= video_len - 1:
                break
            try:
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            bbox = self.coord_list_cycle[self.idx % (len(self.coord_list_cycle))]
            ori_frame = copy.deepcopy(
                self.frame_list_cycle[self.idx % (len(self.frame_list_cycle))]
            )
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except Exception:
                continue
            mask = self.mask_list_cycle[self.idx % (len(self.mask_list_cycle))]
            mask_crop_box = self.mask_coords_list_cycle[
                self.idx % (len(self.mask_coords_list_cycle))
            ]
            combine_frame = get_image_blending(
                ori_frame, res_frame, bbox, mask, mask_crop_box
            )

            if skip_save_images is False:
                cv2.imwrite(
                    f"{self.avatar_path}/tmp/{str(self.idx).zfill(8)}.png",
                    combine_frame,
                )
            self.idx = self.idx + 1

    @torch.no_grad()
    def inference(
        self, audio_path: str, out_vid_name: str, fps: int, skip_save_images: bool
    ) -> str | None:
        os.makedirs(self.avatar_path + "/tmp", exist_ok=True)
        print("start realtime inference")
        whisper_input_features, librosa_length = self.ctx.audio_processor.get_audio_feature(
            audio_path, weight_dtype=self.ctx.weight_dtype
        )
        whisper_chunks = self.ctx.audio_processor.get_whisper_chunk(
            whisper_input_features,
            self.ctx.device,
            self.ctx.weight_dtype,
            self.ctx.whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=self.ctx.audio_padding_length_left,
            audio_padding_length_right=self.ctx.audio_padding_length_right,
        )
        video_num = len(whisper_chunks)
        res_frame_queue: queue.Queue[Any] = queue.Queue()
        self.idx = 0
        process_thread = threading.Thread(
            target=self.process_frames,
            args=(res_frame_queue, video_num, skip_save_images),
        )
        process_thread.start()

        gen = datagen(
            whisper_chunks,
            self.input_latent_list_cycle,
            self.batch_size,
            device=str(self.ctx.device),
        )
        t0 = time.time()
        for whisper_batch, latent_batch in tqdm(
            gen, total=int(np.ceil(float(video_num) / self.batch_size))
        ):
            audio_feature_batch = self.ctx.pe(whisper_batch.to(self.ctx.device))
            latent_batch = latent_batch.to(
                device=self.ctx.device, dtype=self.ctx.unet.model.dtype
            )

            pred_latents = self.ctx.unet.model(
                latent_batch,
                self.ctx.timesteps,
                encoder_hidden_states=audio_feature_batch,
            ).sample
            pred_latents = pred_latents.to(
                device=self.ctx.device, dtype=self.ctx.vae.vae.dtype
            )
            recon = self.ctx.vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_queue.put(res_frame)
        process_thread.join()
        print(
            f"realtime inference {video_num} frames in {time.time() - t0:.2f}s "
            f"(skip_save_images={skip_save_images})"
        )

        if out_vid_name is None or skip_save_images:
            return None

        tmp_glob = os.path.join(self.avatar_path, "tmp", "%08d.png")
        temp_mp4 = os.path.join(self.avatar_path, "temp.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "warning",
                "-r",
                str(fps),
                "-f",
                "image2",
                "-i",
                tmp_glob,
                "-vcodec",
                "libx264",
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
                "-crf",
                "18",
                temp_mp4,
            ],
            check=True,
        )
        os.makedirs(self.video_out_path, exist_ok=True)
        output_vid = os.path.join(self.video_out_path, out_vid_name + ".mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "warning",
                "-i",
                audio_path,
                "-i",
                temp_mp4,
                output_vid,
            ],
            check=True,
        )
        os.remove(temp_mp4)
        shutil.rmtree(os.path.join(self.avatar_path, "tmp"), ignore_errors=True)
        if self.upscale_target_wh is not None:
            uw, uh = self.upscale_target_wh
            tmp_up = os.path.join(self.avatar_path, "upscaled_result.mp4")
            upscale_video_replace_audio(output_vid, audio_path, uw, uh, tmp_up)
            os.replace(tmp_up, output_vid)
            print(f"realtime result upscaled to {uw}x{uh} -> {output_vid}", flush=True)
        print(f"realtime result saved to {output_vid}")
        return output_vid

    @classmethod
    def from_prepared(
        cls,
        ctx: RealtimeJobContext,
        avatar_root: str,
        batch_size: int,
        avatar_id: str,
    ) -> RealtimeAvatar:
        """Load a previously prepared avatar from disk (no landmark/mask prep)."""
        self = object.__new__(cls)
        self.ctx = ctx
        self.avatar_id = avatar_id
        self.video_path = ""
        self.bbox_shift = 0.0
        self.base_path = avatar_root
        self.avatar_path = self.base_path
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.batch_size = batch_size
        self.idx = 0

        if not os.path.isfile(self.latents_out_path):
            raise FileNotFoundError(f"missing prepared avatar: {self.latents_out_path}")
        self.input_latent_list_cycle = torch.load(self.latents_out_path)
        with open(self.coords_path, "rb") as f:
            self.coord_list_cycle = pickle.load(f)
        input_img_list = sorted(
            glob.glob(os.path.join(self.full_imgs_path, "*.[jpJP][pnPN]*[gG]")),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
        )
        if not input_img_list:
            raise FileNotFoundError(f"no frames under {self.full_imgs_path}")
        self.frame_list_cycle = read_imgs(input_img_list)
        with open(self.mask_coords_path, "rb") as f:
            self.mask_coords_list_cycle = pickle.load(f)
        input_mask_list = sorted(
            glob.glob(os.path.join(self.mask_out_path, "*.[jpJP][pnPN]*[gG]")),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
        )
        self.mask_list_cycle = read_imgs(input_mask_list)
        if os.path.isfile(self.avatar_info_path):
            with open(self.avatar_info_path) as f:
                self.avatar_info = json.load(f)
        else:
            self.avatar_info = {"avatar_id": avatar_id, "version": ctx.version}
        uw = self.avatar_info.get("upscale_width")
        uh = self.avatar_info.get("upscale_height")
        try:
            if uw is not None and uh is not None:
                iw, ih = int(uw), int(uh)
                self.upscale_target_wh = (iw, ih) if iw > 1 and ih > 1 else None
            else:
                self.upscale_target_wh = None
        except (TypeError, ValueError):
            self.upscale_target_wh = None
        return self


def _avatar_store_dir() -> str:
    root = Path(os.environ.get("API_JOB_DIR", "./results/api_jobs"))
    d = root / "realtime_avatars"
    d.mkdir(parents=True, exist_ok=True)
    return str(d.resolve())


def run_realtime_job(
    ctx: RealtimeJobContext,
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
) -> tuple[str, str, str]:
    """
    If ``reuse_avatar`` is False: extract first ``prep_frames`` from ``video_path``,
    build avatar under shared ``realtime_avatars/{persist_avatar_id}/``, infer, copy
    result to ``work_dir/result.mp4``.

    If True: load existing ``realtime_avatars/{persist_avatar_id}/``, infer only,
    copy result to ``work_dir/result.mp4``.

    Returns ``(result_path, info, persist_avatar_id)``.
    """
    ctx = RealtimeJobContext(
        version=ctx.version,
        extra_margin=extra_margin,
        parsing_mode=parsing_mode,
        audio_padding_length_left=ctx.audio_padding_length_left,
        audio_padding_length_right=ctx.audio_padding_length_right,
        skip_save_images=ctx.skip_save_images,
        left_cheek_width=left_cheek_width,
        right_cheek_width=right_cheek_width,
        vae=ctx.vae,
        unet=ctx.unet,
        pe=ctx.pe,
        whisper=ctx.whisper,
        audio_processor=ctx.audio_processor,
        fp=FaceParsing(
            left_cheek_width=left_cheek_width,
            right_cheek_width=right_cheek_width,
        ),
        device=ctx.device,
        weight_dtype=ctx.weight_dtype,
        timesteps=ctx.timesteps,
    )

    store = _avatar_store_dir()
    avatar_root = os.path.join(store, persist_avatar_id)
    out_vid_base = "j" + job_id.replace("-", "")
    try:
        scale = parse_resolution_scale(resolution_scale)
    except ValueError as e:
        raise RuntimeError(str(e)) from e

    if reuse_avatar:
        avatar = RealtimeAvatar.from_prepared(
            ctx, avatar_root, batch_size, persist_avatar_id
        )
        info = (
            f"realtime_reuse; avatar_id={persist_avatar_id}; "
            f"parsing_mode={parsing_mode}"
        )
    else:
        if not video_path or not os.path.isfile(video_path):
            raise FileNotFoundError("video_path required for new avatar preparation")
        frames_dir = os.path.join(work_dir, "prep_frames")
        extract_first_frames_ffmpeg(video_path, frames_dir, prep_frames)
        upscale_wh: tuple[int, int] | None = None
        if scale < 1.0 - 1e-9:
            upscale_wh = downscale_png_dir_inplace(frames_dir, scale)

        avatar = RealtimeAvatar(
            ctx,
            avatar_id=persist_avatar_id,
            video_path=frames_dir,
            bbox_shift=bbox_shift,
            batch_size=batch_size,
            avatar_root=avatar_root,
            upscale_target_wh=upscale_wh,
        )
        info = (
            f"realtime; avatar_id={persist_avatar_id}; prep_frames={prep_frames}; "
            f"bbox_shift={bbox_shift}; parsing_mode={parsing_mode}"
        )

    out = avatar.inference(
        audio_path, out_vid_base, fps=fps, skip_save_images=ctx.skip_save_images
    )
    if not out:
        raise RuntimeError("realtime inference produced no video (check skip_save_images)")

    result_copy = os.path.join(work_dir, "result.mp4")
    shutil.copy2(out, result_copy)
    return result_copy, info, persist_avatar_id
