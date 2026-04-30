import os
import re
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import jwt
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import ExpiredSignatureError, InvalidTokenError

from musetalk.service.config import ServiceConfig, load_service_config
from musetalk.service.voice_cloner_mount import mount_voice_cloner_if_enabled
from musetalk.service.ffmpeg_pipe import has_nvenc_encoder, has_working_nvenc
from musetalk.service.mux_demux import demux_muxed_mp4
from musetalk.service.resolution_scale import parse_resolution_scale

_JOBS_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}

security = HTTPBearer(auto_error=False)


def _job_root() -> Path:
    root = Path(os.environ.get("API_JOB_DIR", "./results/api_jobs"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _valid_contract_job_id(job_id: str) -> bool:
    if not job_id or len(job_id) > 256:
        return False
    return bool(re.match(r"^[a-zA-Z0-9._-]+$", job_id))


def _require_bearer(
    cfg: ServiceConfig,
    creds: HTTPAuthorizationCredentials | None = Depends(security),
) -> None:
    _resolve_user_id(cfg, creds)


def _resolve_user_id(
    cfg: ServiceConfig,
    creds: HTTPAuthorizationCredentials | None,
) -> str:
    """
    Auth rules:
    1) Admin override: Authorization Bearer == BEARER_TOKEN.
    2) Otherwise, decode JWT using JWT_SECRET / JWT_ALGORITHM and use sub|uid as user_id.
    """
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token")
    token = (creds.credentials or "").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token")

    if cfg.bearer_token and token == cfg.bearer_token:
        # Admin path still enabled via BEARER_TOKEN.
        return "admin"

    if not cfg.jwt_secret:
        raise HTTPException(
            status_code=401,
            detail="Invalid bearer token (JWT auth not configured)",
        )
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            cfg.jwt_secret,
            algorithms=[cfg.jwt_algorithm],
            options={"require": ["exp"]},
        )
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid bearer token")

    user_id = str(payload.get("sub") or payload.get("uid") or "").strip()
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing uid/sub claim")
    return user_id


def create_service_app(
    inference_fn: Callable[..., tuple[str, str]],
    config: ServiceConfig | None = None,
    realtime_runner: Callable[..., tuple[str, str, str]] | None = None,
) -> FastAPI:
    cfg = config or load_service_config()
    nvenc_available = has_nvenc_encoder()
    nvenc_working = has_working_nvenc()
    requested_encoder = (cfg.ffmpeg_video_encoder or "").strip() or "h264_nvenc"
    effective_encoder = (
        "libx264"
        if requested_encoder.endswith("_nvenc") and not nvenc_working
        else requested_encoder
    )
    print(
        "[service-config] "
        f"cpu_workers={cfg.cpu_workers} "
        f"parallel_blend={cfg.enable_parallel_blend} "
        f"audio_frame_overlap={cfg.enable_parallel_audio_frame_overlap} "
        f"parallel_realtime_prep={cfg.enable_parallel_realtime_prep} "
        f"standard_batch_size={cfg.standard_batch_size} "
        f"realtime_batch_size_default={cfg.realtime_batch_size_default} "
        f"landmark_batch_size={cfg.landmark_batch_size} "
        f"ffmpeg_encoder_requested={requested_encoder} "
        f"ffmpeg_encoder_effective={effective_encoder} "
        f"ffmpeg_nvenc_available={nvenc_available} "
        f"ffmpeg_nvenc_working={nvenc_working}",
        flush=True,
    )

    def bearer_dep(
        creds: HTTPAuthorizationCredentials | None = Depends(security),
    ) -> None:
        _require_bearer(cfg, creds)

    def auth_user_dep(
        creds: HTTPAuthorizationCredentials | None = Depends(security),
    ) -> str:
        return _resolve_user_id(cfg, creds)

    app = FastAPI(title="MuseTalk Service", version="1.0")
    app.state.voice_cloner_mounted = mount_voice_cloner_if_enabled(
        app, enabled=bool(cfg.enable_voice_cloner)
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        out: dict[str, Any] = {
            "status": "ok",
            "cpu_workers": cfg.cpu_workers,
            "parallel_blend": cfg.enable_parallel_blend,
            "audio_frame_overlap": cfg.enable_parallel_audio_frame_overlap,
            "parallel_realtime_prep": cfg.enable_parallel_realtime_prep,
            "streaming_standard": cfg.enable_streaming_standard,
            "streaming_realtime": cfg.enable_streaming_realtime,
            "streaming_pipe_buffer_frames": cfg.streaming_pipe_buffer_frames,
            "ffmpeg_video_encoder": cfg.ffmpeg_video_encoder,
            "ffmpeg_encoder_preset": cfg.ffmpeg_encoder_preset,
            "ffmpeg_encoder_crf": cfg.ffmpeg_encoder_crf,
            "ffmpeg_encoder_cq": cfg.ffmpeg_encoder_cq,
            "ffmpeg_use_gpu_scale": cfg.ffmpeg_use_gpu_scale,
            "standard_batch_size": cfg.standard_batch_size,
            "realtime_batch_size_default": cfg.realtime_batch_size_default,
            "landmark_batch_size": cfg.landmark_batch_size,
            "voice_cloner": bool(getattr(app.state, "voice_cloner_mounted", False)),
            "voice_cloner_prefix": "/api/voice" if getattr(app.state, "voice_cloner_mounted", False) else None,
        }
        if getattr(app.state, "voice_cloner_mounted", False):
            ddb_user = bool(
                os.environ.get("DDB_TABLE_NAME", "").strip()
                and os.environ.get("AWS_REGION", "").strip()
            )
            out["voice_cloner_auth"] = "ddb" if ddb_user else "jwt"
            region = os.environ.get("AWS_REGION", "").strip()
            emb = bool(region and os.environ.get("EMBEDDINGS_TABLE_NAME", "").strip())
            s3 = bool(region and os.environ.get("S3_BUCKET", "").strip())
            out["voice_cloner_backend"] = "aws" if emb and s3 else "local"
        return out

    @app.post("/job", dependencies=[Depends(bearer_dep)])
    async def contract_post_job(
        request: Request,
        job_id: str = Query(..., alias="id"),
    ) -> JSONResponse:
        if not _valid_contract_job_id(job_id):
            raise HTTPException(status_code=400, detail="invalid job id")
        with _JOBS_LOCK:
            if job_id in _JOBS:
                st = _JOBS[job_id]["status"]
                if st in ("queued", "processing"):
                    raise HTTPException(
                        status_code=409, detail="job already in progress"
                    )
                if st == "done":
                    raise HTTPException(
                        status_code=409,
                        detail="job already completed; use a new id",
                    )
        work = _job_root() / "contract" / job_id
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)
        muxed = work / "muxed_input.mp4"

        form = await request.form()
        field = cfg.gpu_multipart_field
        item = form.get(field)
        if item is None:
            raise HTTPException(
                status_code=400,
                detail=f"missing multipart field {field!r} (set GPU_MULTIPART_FIELD if needed)",
            )
        if hasattr(item, "read"):
            raw = await item.read()
        else:
            raise HTTPException(
                status_code=400, detail=f"field {field!r} must be a file upload"
            )
        if not raw:
            raise HTTPException(status_code=400, detail="empty upload")
        with open(muxed, "wb") as mf:
            mf.write(raw)

        with _JOBS_LOCK:
            _JOBS[job_id] = {
                "status": "queued",
                "work_dir": str(work),
                "message": "",
                "kind": "contract",
                "cpu_workers": cfg.cpu_workers,
            }
        threading.Thread(
            target=_run_contract_muxed_job,
            args=(job_id, str(muxed), str(work), inference_fn),
            daemon=True,
        ).start()
        return JSONResponse(
            status_code=202, content={"status": "queued", "id": job_id}
        )

    @app.get("/job", dependencies=[Depends(bearer_dep)])
    def contract_get_job(job_id: str = Query(..., alias="id")):
        with _JOBS_LOCK:
            info = _JOBS.get(job_id)
        if not info:
            raise HTTPException(status_code=404, detail="unknown job id")
        st = info["status"]
        if st in ("queued", "processing"):
            return JSONResponse(content={"status": "processing"})
        if st == "error":
            return JSONResponse(
                content={
                    "status": "error",
                    "message": info.get("message", ""),
                }
            )
        if st == "done":
            path = info.get("result_path")
            if not path or not os.path.isfile(path):
                raise HTTPException(status_code=500, detail="Result file missing")
            return FileResponse(
                path,
                media_type="application/octet-stream",
                filename=os.path.basename(path),
            )
        return JSONResponse(content={"status": st})

    @app.post("/api/job", dependencies=[Depends(bearer_dep)])
    async def create_job(
        audio: UploadFile = File(..., description="Driving audio"),
        video: UploadFile = File(..., description="Reference video"),
        bbox_shift: float = Form(0),
        extra_margin: int = Form(10),
        parsing_mode: str = Form("jaw"),
        left_cheek_width: int = Form(90),
        right_cheek_width: int = Form(90),
        resolution_scale: str = Form(
            "full",
            description="Process at reduced resolution then upscale: full, half, eighth, lowest",
        ),
    ) -> JSONResponse:
        t_up = time.perf_counter()
        try:
            parse_resolution_scale(resolution_scale)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        job_id = str(uuid.uuid4())
        work = _job_root() / job_id
        work.mkdir(parents=True, exist_ok=False)

        audio_suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
        video_suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
        audio_path = work / f"input_audio{audio_suffix}"
        video_path = work / f"input_video{video_suffix}"

        with open(audio_path, "wb") as af:
            shutil.copyfileobj(audio.file, af)
        with open(video_path, "wb") as vf:
            shutil.copyfileobj(video.file, vf)
        upload_elapsed = time.perf_counter() - t_up

        with _JOBS_LOCK:
            _JOBS[job_id] = {
                "status": "queued",
                "work_dir": str(work),
                "message": f"upload {upload_elapsed:.2f}s",
                "kind": "standard",
                "cpu_workers": cfg.cpu_workers,
                "stage_times": {"upload": round(upload_elapsed, 3)},
                "pipeline_mode": (
                    "streaming_standard"
                    if cfg.enable_streaming_standard
                    else "legacy"
                ),
            }

        threading.Thread(
            target=_run_job_background,
            args=(
                job_id,
                str(audio_path),
                str(video_path),
                bbox_shift,
                extra_margin,
                parsing_mode,
                left_cheek_width,
                right_cheek_width,
                inference_fn,
                resolution_scale,
            ),
            daemon=True,
        ).start()
        return JSONResponse(
            status_code=202,
            content={"job_id": job_id, "status": "queued"},
        )

    if realtime_runner is not None:

        @app.post("/api/realtime/job", dependencies=[Depends(bearer_dep)])
        async def create_realtime_job(
            audio: UploadFile = File(..., description="Driving audio"),
            video: Optional[UploadFile] = File(
                None, description="Reference video (required unless use_clone=true)"
            ),
            bbox_shift: float = Form(0),
            extra_margin: int = Form(10),
            parsing_mode: str = Form("jaw"),
            left_cheek_width: int = Form(90),
            right_cheek_width: int = Form(90),
            realtime_prep_frames: int = Form(30),
            realtime_batch_size: int = Form(cfg.realtime_batch_size_default),
            realtime_fps: int = Form(25),
            clone_id: Optional[str] = Form(
                None,
                description="Optional clone/user id. If null, uses JWT uid/sub.",
            ),
            use_clone: bool = Form(
                False,
                description="If true, skip video prep and reuse persisted clone materials.",
            ),
            resolution_scale: str = Form(
                "full",
                description="Prep/infer at reduced res then upscale: full, half, eighth, lowest",
            ),
            user_id: str = Depends(auth_user_dep),
        ) -> JSONResponse:
            t_up = time.perf_counter()
            try:
                parse_resolution_scale(resolution_scale)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            prep = max(1, min(300, int(realtime_prep_frames)))
            bs = max(1, min(128, int(realtime_batch_size)))
            fps = max(1, min(60, int(realtime_fps)))

            requested_clone = (clone_id or "").strip()
            if requested_clone and not _valid_contract_job_id(requested_clone):
                raise HTTPException(status_code=400, detail="invalid clone_id")
            persist_user_id = requested_clone or user_id
            if not _valid_contract_job_id(persist_user_id):
                raise HTTPException(status_code=400, detail="invalid resolved user_id")

            if use_clone:
                av_dir = _job_root() / "realtime_avatars" / persist_user_id
                if not (av_dir / "latents.pt").is_file():
                    raise HTTPException(
                        status_code=404,
                        detail=f"unknown or incomplete clone_id/user_id: {persist_user_id!r}",
                    )
                is_reuse = True
                video_disk_path: str | None = None
            else:
                if video is None:
                    raise HTTPException(
                        status_code=400,
                        detail="video is required unless use_clone is true",
                    )
                is_reuse = False
                video_suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
                video_disk_path = f"input_video{video_suffix}"

            job_id = str(uuid.uuid4())
            work = _job_root() / "realtime" / job_id
            work.mkdir(parents=True, exist_ok=False)

            audio_suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
            audio_path = work / f"input_audio{audio_suffix}"

            with open(audio_path, "wb") as af:
                shutil.copyfileobj(audio.file, af)

            video_str = ""
            if not is_reuse and video_disk_path is not None and video is not None:
                vp = work / video_disk_path
                with open(vp, "wb") as vf:
                    shutil.copyfileobj(video.file, vf)
                video_str = str(vp)
            upload_elapsed = time.perf_counter() - t_up

            with _JOBS_LOCK:
                _JOBS[job_id] = {
                    "status": "queued",
                    "work_dir": str(work),
                    "message": f"upload {upload_elapsed:.2f}s",
                    "kind": "realtime",
                    "realtime_prep_frames": prep,
                    "user_id": persist_user_id,
                    "clone_id": persist_user_id,
                    "reuse_avatar": is_reuse,
                    "use_clone": is_reuse,
                    "cpu_workers": cfg.cpu_workers,
                    "stage_times": {"upload": round(upload_elapsed, 3)},
                    "pipeline_mode": (
                        "streaming_realtime"
                        if cfg.enable_streaming_realtime
                        else "legacy"
                    ),
                }

            threading.Thread(
                target=_run_realtime_job_background,
                args=(
                    job_id,
                    str(audio_path),
                    video_str,
                    str(work),
                    persist_user_id,
                    is_reuse,
                    bbox_shift,
                    extra_margin,
                    parsing_mode,
                    left_cheek_width,
                    right_cheek_width,
                    prep,
                    bs,
                    fps,
                    realtime_runner,
                    resolution_scale,
                ),
                daemon=True,
            ).start()
            return JSONResponse(
                status_code=202,
                content={
                    "job_id": job_id,
                    "user_id": persist_user_id,
                    "clone_id": persist_user_id,
                    "status": "queued",
                    "kind": "realtime",
                    "realtime_prep_frames": prep,
                    "use_clone": is_reuse,
                },
            )

    @app.get("/api/job/{job_id}", dependencies=[Depends(bearer_dep)])
    def job_status(job_id: str) -> dict[str, Any]:
        with _JOBS_LOCK:
            info = _JOBS.get(job_id)
        if not info:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        out: dict[str, Any] = {
            "job_id": job_id,
            "status": info["status"],
            "message": info.get("message", ""),
        }
        if info.get("stage_times"):
            out["stage_times"] = info["stage_times"]
        if info.get("pipeline_mode"):
            out["pipeline_mode"] = info["pipeline_mode"]
        if info.get("kind"):
            out["kind"] = info["kind"]
        if info.get("user_id"):
            out["user_id"] = info["user_id"]
        if info.get("clone_id"):
            out["clone_id"] = info["clone_id"]
        if info["status"] == "done":
            out["bbox_shift_text"] = info.get("bbox_shift_text", "")
        return out

    @app.get(
        "/api/job/{job_id}/download",
        dependencies=[Depends(bearer_dep)],
    )
    def job_download(job_id: str):
        with _JOBS_LOCK:
            info = _JOBS.get(job_id)
        if not info:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        if info["status"] != "done":
            raise HTTPException(
                status_code=409,
                detail=f"Job not ready (status={info['status']})",
            )
        path = info.get("result_path")
        if not path or not os.path.isfile(path):
            raise HTTPException(status_code=500, detail="Result file missing")
        return FileResponse(
            path,
            media_type="video/mp4",
            filename=os.path.basename(path),
        )

    return app


def _run_contract_muxed_job(
    job_id: str,
    muxed_path: str,
    work_dir: str,
    inference_fn: Callable[..., tuple[str, str]],
) -> None:
    def set_status(status: str, **extra: Any) -> None:
        with _JOBS_LOCK:
            if job_id in _JOBS:
                _JOBS[job_id]["status"] = status
                for k, v in extra.items():
                    _JOBS[job_id][k] = v

    try:
        set_status("processing")
        t0 = time.perf_counter()
        vpath, apath = demux_muxed_mp4(muxed_path, Path(work_dir))
        t1 = time.perf_counter()
        print(
            f"[job {job_id}] contract demux_muxed_mp4 dt={t1 - t0:.3f}s",
            flush=True,
        )
        out_path, bbox_text = inference_fn(
            apath,
            vpath,
            0,
            10,
            "jaw",
            90,
            90,
        )
        t2 = time.perf_counter()
        print(
            f"[job {job_id}] contract inference_fn dt={t2 - t1:.3f}s "
            f"total={t2 - t0:.3f}s",
            flush=True,
        )
        set_status(
            "done",
            result_path=out_path,
            bbox_shift_text=bbox_text,
            message="",
        )
    except Exception as e:
        print(f"[job {job_id}] contract mux job failed: {e}", flush=True)
        traceback.print_exc()
        set_status("error", message=str(e))


def _run_job_background(
    job_id: str,
    audio_path: str,
    video_path: str,
    bbox_shift: float,
    extra_margin: int,
    parsing_mode: str,
    left_cheek_width: int,
    right_cheek_width: int,
    inference_fn: Callable[..., tuple[str, str]],
    resolution_scale: str,
) -> None:
    def set_status(status: str, **extra: Any) -> None:
        with _JOBS_LOCK:
            if job_id in _JOBS:
                _JOBS[job_id]["status"] = status
                for k, v in extra.items():
                    _JOBS[job_id][k] = v

    try:
        print(
            f"[job {job_id}] /api/job worker started "
            f"(audio={audio_path}, video={video_path})",
            flush=True,
        )
        set_status("processing", message="worker started")
        print(f"[job {job_id}] /api/job entering inference()", flush=True)
        with _JOBS_LOCK:
            initial = dict((_JOBS.get(job_id) or {}).get("stage_times") or {})
        stage_times: dict[str, float] = {k: float(v) for k, v in initial.items()}

        def stage_callback(stage: str, elapsed: float, snapshot: dict[str, float]) -> None:
            stage_times.update(snapshot)
            set_status(
                "processing",
                stage=stage,
                message=f"{stage} {elapsed:.2f}s",
                stage_times={k: round(v, 3) for k, v in stage_times.items()},
            )

        set_status("processing", stage="preprocess", message="preprocess 0.00s")
        t_inf = time.perf_counter()
        out_path, bbox_text = inference_fn(
            audio_path,
            video_path,
            bbox_shift,
            extra_margin,
            parsing_mode,
            left_cheek_width,
            right_cheek_width,
            resolution_scale=resolution_scale,
            status_callback=stage_callback,
        )
        print(
            f"[job {job_id}] /api/job inference_fn wall={time.perf_counter() - t_inf:.3f}s "
            f"out={out_path}",
            flush=True,
        )
        set_status(
            "done",
            result_path=out_path,
            bbox_shift_text=bbox_text,
            message="",
            stage="done",
            stage_times={k: round(v, 3) for k, v in stage_times.items()},
        )
    except Exception as e:
        print(f"[job {job_id}] /api/job failed: {e}", flush=True)
        traceback.print_exc()
        set_status("error", message=str(e))


def _run_realtime_job_background(
    job_id: str,
    audio_path: str,
    video_path: str,
    work_dir: str,
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
    realtime_runner: Callable[..., tuple[str, str, str]],
    resolution_scale: str,
) -> None:
    def set_status(status: str, **extra: Any) -> None:
        with _JOBS_LOCK:
            if job_id in _JOBS:
                _JOBS[job_id]["status"] = status
                for k, v in extra.items():
                    _JOBS[job_id][k] = v

    try:
        print(
            f"[job {job_id}] /api/realtime/job worker started "
            f"reuse_avatar={reuse_avatar} work_dir={work_dir}",
            flush=True,
        )
        with _JOBS_LOCK:
            initial = dict((_JOBS.get(job_id) or {}).get("stage_times") or {})
        stage_times: dict[str, float] = {k: float(v) for k, v in initial.items()}

        def stage_callback(stage: str, elapsed: float, snapshot: dict[str, float]) -> None:
            stage_times.update(snapshot)
            set_status(
                "processing",
                stage=stage,
                message=f"{stage} {elapsed:.2f}s",
                stage_times={k: round(v, 3) for k, v in stage_times.items()},
            )

        set_status("processing", stage="preprocess", message="preprocess 0.00s")
        t_rt = time.perf_counter()
        out_path, bbox_text, _aid = realtime_runner(
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
            status_callback=stage_callback,
        )
        print(
            f"[job {job_id}] /api/realtime/job realtime_runner wall="
            f"{time.perf_counter() - t_rt:.3f}s out={out_path}",
            flush=True,
        )
        set_status(
            "done",
            result_path=out_path,
            bbox_shift_text=bbox_text,
            message="",
            user_id=persist_avatar_id,
            clone_id=persist_avatar_id,
            stage="done",
            stage_times={k: round(v, 3) for k, v in stage_times.items()},
        )
    except Exception as e:
        print(f"[job {job_id}] /api/realtime/job failed: {e}", flush=True)
        traceback.print_exc()
        set_status("error", message=str(e))
