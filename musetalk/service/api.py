import os
import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from musetalk.service.config import ServiceConfig, load_service_config
from musetalk.service.mux_demux import demux_muxed_mp4

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
    if not cfg.bearer_token:
        return
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token")
    if creds.credentials != cfg.bearer_token:
        raise HTTPException(status_code=401, detail="Invalid bearer token")


def create_service_app(
    inference_fn: Callable[..., tuple[str, str]],
    config: ServiceConfig | None = None,
) -> FastAPI:
    cfg = config or load_service_config()

    def bearer_dep(
        creds: HTTPAuthorizationCredentials | None = Depends(security),
    ) -> None:
        _require_bearer(cfg, creds)

    app = FastAPI(title="MuseTalk Service", version="1.0")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

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
    ) -> JSONResponse:
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

        with _JOBS_LOCK:
            _JOBS[job_id] = {
                "status": "queued",
                "work_dir": str(work),
                "message": "",
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
            ),
            daemon=True,
        ).start()
        return JSONResponse(
            status_code=202,
            content={"job_id": job_id, "status": "queued"},
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
        vpath, apath = demux_muxed_mp4(muxed_path, Path(work_dir))
        out_path, bbox_text = inference_fn(
            apath,
            vpath,
            0,
            10,
            "jaw",
            90,
            90,
        )
        set_status(
            "done",
            result_path=out_path,
            bbox_shift_text=bbox_text,
            message="",
        )
    except Exception as e:
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
) -> None:
    def set_status(status: str, **extra: Any) -> None:
        with _JOBS_LOCK:
            if job_id in _JOBS:
                _JOBS[job_id]["status"] = status
                for k, v in extra.items():
                    _JOBS[job_id][k] = v

    try:
        set_status("processing")
        out_path, bbox_text = inference_fn(
            audio_path,
            video_path,
            bbox_shift,
            extra_margin,
            parsing_mode,
            left_cheek_width,
            right_cheek_width,
        )
        set_status(
            "done",
            result_path=out_path,
            bbox_shift_text=bbox_text,
            message="",
        )
    except Exception as e:
        set_status("error", message=str(e))
