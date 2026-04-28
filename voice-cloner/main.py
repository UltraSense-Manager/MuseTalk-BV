# Disable torchaudio's FFmpeg backend before torch is loaded. Torchaudio ships an FFmpeg 6
# extension (libavutil.58); system FFmpeg 7/8 use different libs, so the extension fails to load.
# With this set, torchaudio uses other backends (e.g. sox/soundfile); OpenVoice uses librosa/pydub anyway.
import os
os.environ.setdefault("TORIO_USE_FFMPEG", "0")

import json
import logging
from typing import Any, Dict, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware

from auth import extract_token_from_websocket, get_current_user, verify_jwt_and_load_user
from audio import (
    append_pcm_chunk,
    compute_trained_voice_id,
    safe_remove,
    safe_rmtree,
    write_b64_audio_to_temp_wav,
)
from cloner import clone as run_clone, extract_speaker_embedding
from log_config import configure_logging
from models import (
    AuthedUser,
    CloneRequest,
    CloneResponse,
    HealthResponse,
    StateResponse,
    TrainChunkResponse,
    TrainRequest,
    TrainResponse,
    TrainStartResponse,
)

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Cloner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory session store keyed by user.sub
SessionData = Dict[str, Any]
sessions: Dict[str, SessionData] = {}


def get_or_create_session(sub: str) -> SessionData:
    if sub not in sessions:
        sessions[sub] = {}
        logger.debug("created new session for sub=%s", sub)
    return sessions[sub]


def _clear_training_buffer(session: SessionData) -> None:
    """Clear reference buffer and training state; keep trained_voice_id etc. until next successful train."""
    session.pop("reference_b64", None)
    session.pop("reference_path", None)
    session.pop("reference_target_se", None)
    session.pop("reference_audio_name", None)
    session["training_in_progress"] = True


def _run_training_from_buffer(session: SessionData, sub: str) -> str:
    """
    Use session["reference_b64"] to write WAV, extract embedding, store in session.
    Returns trained_voice_id. Caller must ensure reference_b64 is set and training_in_progress was True.
    """
    combined_b64 = session.get("reference_b64")
    if not combined_b64:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No reference audio received; send 'start' then 'reference' chunks then 'end'.",
        )
    voice_id = compute_trained_voice_id(combined_b64)
    session["trained_voice_id"] = voice_id
    reference_path = write_b64_audio_to_temp_wav(prefix=f"ref_{sub}", audio_b64=combined_b64)
    audio_name: Optional[str] = None
    try:
        target_se, audio_name = extract_speaker_embedding(reference_path)
        session["reference_target_se"] = target_se
        session["reference_audio_name"] = audio_name
        session.pop("training_in_progress", None)
        return voice_id
    finally:
        safe_remove(reference_path)
        if audio_name:
            safe_rmtree(os.path.join("processed", audio_name))


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    logger.debug("GET /health")
    return HealthResponse(status="ok")


@app.get("/state", response_model=StateResponse)
async def get_state(current_user: AuthedUser = Depends(get_current_user)) -> StateResponse:
    logger.info("GET /state sub=%s", current_user.sub)
    session = get_or_create_session(current_user.sub)
    trained_voice_id = session.get("trained_voice_id")
    logger.debug("state sub=%s trained_voice_id=%s", current_user.sub, trained_voice_id)
    return StateResponse(session_id=current_user.sub, trained_voice_id=trained_voice_id)


@app.post(
    "/train",
    response_model=Union[TrainStartResponse, TrainChunkResponse, TrainResponse],
)
async def train_voice(
    body: TrainRequest, current_user: AuthedUser = Depends(get_current_user)
) -> Union[TrainStartResponse, TrainChunkResponse, TrainResponse]:
    session = get_or_create_session(current_user.sub)

    if body.operation == "start":
        logger.info("POST /train operation=start sub=%s", current_user.sub)
        _clear_training_buffer(session)
        return TrainStartResponse()

    if body.reference is not None:
        if not session.get("training_in_progress"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Send {'operation': 'start'} first, then {'reference': '...'} chunks, then {'operation': 'end'}.",
            )
        logger.debug("POST /train chunk sub=%s", current_user.sub)
        existing_b64 = session.get("reference_b64")
        combined_b64, is_complete = append_pcm_chunk(existing_b64, body.reference)
        session["reference_b64"] = combined_b64
        logger.debug("train append_pcm_chunk sub=%s is_complete=%s", current_user.sub, is_complete)
        return TrainChunkResponse()

    if body.operation == "end":
        logger.info("POST /train operation=end sub=%s", current_user.sub)
        voice_id = _run_training_from_buffer(session, current_user.sub)
        logger.info("train done sub=%s trained_voice_id=%s", current_user.sub, voice_id)
        return TrainResponse(session_id=current_user.sub, trained_voice_id=voice_id)

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Expected 'operation': 'start' | 'end' or 'reference': base64 chunk.",
    )


@app.post("/clone", response_model=CloneResponse)
async def clone_voice(
    body: CloneRequest, current_user: AuthedUser = Depends(get_current_user)
) -> CloneResponse:
    logger.info("POST /clone sub=%s", current_user.sub)
    session = get_or_create_session(current_user.sub)
    trained_voice_id: Optional[str] = session.get("trained_voice_id")
    target_se = session.get("reference_target_se")
    target_audio_name = session.get("reference_audio_name")

    if not trained_voice_id or target_se is None:
        logger.warning("clone rejected: no trained voice sub=%s", current_user.sub)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No trained voice registered for this session",
        )

    # Base audio is also truncated to 20 seconds if longer
    base_path = write_b64_audio_to_temp_wav(
        prefix=f"base_{current_user.sub}", audio_b64=body.base
    )
    try:
        output_path = run_clone(
            reference_speaker="",
            base_speaker=base_path,
            target_se=target_se,
            target_audio_name=target_audio_name,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("clone failed sub=%s: %s", current_user.sub, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate cloned voice: {exc}",
        ) from exc
    finally:
        safe_remove(base_path)

    logger.info("clone done sub=%s trained_voice_id=%s", current_user.sub, trained_voice_id)
    return CloneResponse(
        session_id=current_user.sub,
        trained_voice_id=trained_voice_id,
        output_path=output_path,
    )


@app.websocket("/train")
async def train_voice_ws(websocket: WebSocket) -> None:
    logger.debug("WS /train connection attempt")
    try:
        token = extract_token_from_websocket(websocket)
        user = verify_jwt_and_load_user(token)
    except HTTPException as exc:
        logger.warning("WS /train auth failed: %s", exc.detail)
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    logger.info("WS /train connected sub=%s", user.sub)
    try:
        while True:
            message = await websocket.receive_text()
            try:
                payload = json.loads(message)
                operation = payload.get("operation")
                reference_b64 = payload.get("reference")
                session = get_or_create_session(user.sub)

                if operation == "start":
                    _clear_training_buffer(session)
                    logger.info("WS /train operation=start sub=%s", user.sub)
                    await websocket.send_json({"type": "started"})
                    continue

                if reference_b64 is not None:
                    if not session.get("training_in_progress"):
                        await websocket.send_json(
                            {"error": "Send {'operation': 'start'} first.", "code": 422}
                        )
                        continue
                    existing_b64 = session.get("reference_b64")
                    combined_b64, is_complete = append_pcm_chunk(existing_b64, reference_b64)
                    session["reference_b64"] = combined_b64
                    logger.debug("WS /train chunk sub=%s is_complete=%s", user.sub, is_complete)
                    await websocket.send_json({"type": "chunk_received"})
                    continue

                if operation == "end":
                    try:
                        voice_id = _run_training_from_buffer(session, user.sub)
                    except HTTPException as e:
                        await websocket.send_json(
                            {"error": e.detail or "Training failed", "code": e.status_code}
                        )
                        continue
                    logger.info("WS /train operation=end done sub=%s trained_voice_id=%s", user.sub, voice_id)
                    await websocket.send_json(
                        {
                            "type": "train_result",
                            "session_id": user.sub,
                            "trained_voice_id": voice_id,
                        }
                    )
                    continue

                await websocket.send_json(
                    {"error": "Expected 'operation': 'start'|'end' or 'reference': base64.", "code": 422}
                )
            except json.JSONDecodeError as e:
                logger.debug("WS /train invalid JSON sub=%s: %s", user.sub, e)
                await websocket.send_json({"error": "Invalid JSON", "code": 400})
    except WebSocketDisconnect:
        logger.info("WS /train disconnected sub=%s", user.sub)


@app.websocket("/clone")
async def clone_voice_ws(websocket: WebSocket) -> None:
    logger.debug("WS /clone connection attempt")
    try:
        token = extract_token_from_websocket(websocket)
        user = verify_jwt_and_load_user(token)
    except HTTPException as exc:
        logger.warning("WS /clone auth failed: %s", exc.detail)
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    logger.info("WS /clone connected sub=%s", user.sub)
    try:
        while True:
            message = await websocket.receive_text()
            try:
                payload = json.loads(message)
                base_b64 = payload.get("base")
                if not base_b64:
                    logger.debug("WS /clone missing base sub=%s", user.sub)
                    await websocket.send_json(
                        {"error": "Missing 'base' field", "code": 422}
                    )
                    continue

                session = get_or_create_session(user.sub)
                trained_voice_id = session.get("trained_voice_id")
                target_se = session.get("reference_target_se")
                target_audio_name = session.get("reference_audio_name")
                if not trained_voice_id or target_se is None:
                    logger.warning("WS /clone no trained voice sub=%s", user.sub)
                    await websocket.send_json(
                        {
                            "error": "No trained voice registered for this session",
                            "code": 422,
                        }
                    )
                    continue

                # Base audio truncated to 20s if longer
                base_path = write_b64_audio_to_temp_wav(
                    prefix=f"base_{user.sub}", audio_b64=base_b64
                )
                try:
                    output_path = run_clone(
                        reference_speaker="",
                        base_speaker=base_path,
                        target_se=target_se,
                        target_audio_name=target_audio_name,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("WS /clone failed sub=%s: %s", user.sub, exc)
                    await websocket.send_json(
                        {
                            "error": f"Failed to generate cloned voice: {exc}",
                            "code": 500,
                        }
                    )
                    continue
                finally:
                    safe_remove(base_path)

                logger.info("WS /clone done sub=%s trained_voice_id=%s", user.sub, trained_voice_id)
                await websocket.send_json(
                    {
                        "type": "clone_result",
                        "session_id": user.sub,
                        "trained_voice_id": trained_voice_id,
                        "output_path": output_path,
                    }
                )
            except json.JSONDecodeError as e:
                logger.debug("WS /clone invalid JSON sub=%s: %s", user.sub, e)
                await websocket.send_json({"error": "Invalid JSON", "code": 400})
    except WebSocketDisconnect:
        logger.info("WS /clone disconnected sub=%s", user.sub)


if __name__ == "__main__":
    import uvicorn

    logger.info("starting uvicorn")
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)

