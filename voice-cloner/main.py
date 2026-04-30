from __future__ import annotations

import os

# Disable torchaudio's FFmpeg backend before torch is loaded. Torchaudio ships an FFmpeg 6
# extension (libavutil.58); system FFmpeg 7/8 use different libs, so the extension fails to load.
# With this set, torchaudio uses other backends (e.g. sox/soundfile); OpenVoice uses librosa/pydub anyway.
os.environ.setdefault("TORIO_USE_FFMPEG", "0")

import base64
import json
import logging
import shutil
import tempfile
import time
from typing import Any, Dict, Optional, Union

import boto3
import numpy as np
import torch
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware

from auth import extract_token_from_websocket, get_current_user, get_current_user_optional, verify_jwt_and_load_user
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
    CloneSubRequest,
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

# Sub → voice link for /clone-sub; expires 1h after first /state for that sub
SUB_LINK_TTL_SEC = 3600
sub_voice_links: Dict[str, Dict[str, Any]] = {}

# In-process embedding rows when DynamoDB embeddings table is not configured (mirrors DDB item shape)
_local_embeddings: Dict[str, Dict[str, Any]] = {}


def _voice_cloner_local_root() -> str:
    root = os.getenv("VOICE_CLONER_LOCAL_DIR", "").strip()
    if not root:
        root = os.path.join(os.getcwd(), "results", "voice_cloner_data")
    os.makedirs(root, exist_ok=True)
    return root


def _save_reference_wav_local(sub: str, trained_voice_id: str, reference_path: str) -> str:
    user_dir = os.path.join(_voice_cloner_local_root(), sub)
    os.makedirs(user_dir, exist_ok=True)
    dest = os.path.join(user_dir, f"{trained_voice_id}.wav")
    shutil.copy2(reference_path, dest)
    logger.debug("Saved reference WAV locally sub=%s path=%s", sub, dest)
    return dest

# DynamoDB table for storing user speaker embeddings
AWS_REGION = os.getenv("AWS_REGION")
EMBEDDINGS_TABLE_NAME = os.getenv("EMBEDDINGS_TABLE_NAME", "brivva-users-embeddings")

embeddings_table = None
if AWS_REGION and EMBEDDINGS_TABLE_NAME:
    try:
        _ddb = boto3.resource("dynamodb", region_name=AWS_REGION)
        embeddings_table = _ddb.Table(EMBEDDINGS_TABLE_NAME)
        logger.info("Embeddings table configured name=%s region=%s", EMBEDDINGS_TABLE_NAME, AWS_REGION)
    except (BotoCoreError, ClientError) as e:  # pragma: no cover - config error path
        logger.exception("Failed to configure embeddings table: %s", e)
        embeddings_table = None

# S3 bucket for reference speaker audio (optional)
S3_BUCKET = os.getenv("S3_BUCKET")
S3_REF_PREFIX = os.getenv("S3_REF_PREFIX", "voice-refs").strip().rstrip("/")
s3_client = None
if AWS_REGION and S3_BUCKET:
    try:
        s3_client = boto3.client("s3", region_name=AWS_REGION)
        logger.info("S3 configured bucket=%s prefix=%s", S3_BUCKET, S3_REF_PREFIX)
    except (BotoCoreError, ClientError) as e:
        logger.warning("S3 client init failed: %s", e)
        s3_client = None


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
    # Keep reference_audio_s3_key until next successful train (overwritten on end)
    session["training_in_progress"] = True


def _store_embedding_for_sub(
    sub: str, trained_voice_id: str, target_se: Any, reference_path: Optional[str] = None
) -> tuple[Optional[str], Optional[str]]:
    """
    Persist speaker embedding (and optionally reference audio).

    - With DynamoDB embeddings table: put_item as before.
    - Without: store same item shape in _local_embeddings[sub].
    - Reference audio: S3 when configured; otherwise copy WAV under VOICE_CLONER_LOCAL_DIR.

    Returns (reference_audio_s3_key_or_none, reference_audio_local_path_or_none).
    """
    reference_audio_s3_key: Optional[str] = None
    reference_audio_local_path: Optional[str] = None
    if reference_path:
        if s3_client and S3_BUCKET:
            try:
                reference_audio_s3_key = f"{S3_REF_PREFIX}/{sub}/{trained_voice_id}.wav"
                with open(reference_path, "rb") as f:
                    s3_client.upload_fileobj(f, S3_BUCKET, reference_audio_s3_key)
                logger.debug("Uploaded reference to s3://%s/%s", S3_BUCKET, reference_audio_s3_key)
            except (BotoCoreError, ClientError, OSError) as e:
                logger.warning("Failed to upload reference to S3 for sub=%s: %s", sub, e)
                reference_audio_s3_key = None
        if not reference_audio_s3_key:
            try:
                reference_audio_local_path = _save_reference_wav_local(
                    sub, trained_voice_id, reference_path
                )
            except OSError as e:
                logger.warning("Failed to save local reference WAV sub=%s: %s", sub, e)
                reference_audio_local_path = None

    try:
        if hasattr(target_se, "cpu"):
            arr = target_se.cpu().numpy()
        elif hasattr(target_se, "__array__"):
            arr = np.asarray(target_se, dtype=np.float32)
        else:
            arr = np.array(target_se, dtype=np.float32)
        shape = list(arr.shape)
        data_b64 = base64.b64encode(arr.astype(np.float32).tobytes()).decode("utf-8")
        item: Dict[str, Any] = {
            "user_id": sub,
            "trained_voice_id": trained_voice_id,
            "embedding_b64": data_b64,
            "embedding_shape": shape,
        }
        if reference_audio_s3_key:
            item["reference_audio_s3_key"] = reference_audio_s3_key
        if reference_audio_local_path:
            item["reference_audio_local_path"] = reference_audio_local_path

        if embeddings_table is not None:
            embeddings_table.put_item(Item=item)
            logger.debug("Stored embedding in DynamoDB sub=%s voice_id=%s", sub, trained_voice_id)
        else:
            _local_embeddings[sub] = item
            logger.debug("Stored embedding in local memory sub=%s voice_id=%s", sub, trained_voice_id)
        return reference_audio_s3_key, reference_audio_local_path
    except (BotoCoreError, ClientError, TypeError, ValueError) as e:
        logger.warning("Failed to store embedding for sub=%s: %s", sub, e)
        return reference_audio_s3_key, reference_audio_local_path


def _apply_embedding_item_to_session(session: SessionData, item: Dict[str, Any]) -> None:
    """Populate session from a DynamoDB-style embedding item."""
    tv_id = item.get("trained_voice_id")

    data_b64 = item.get("embedding_b64")
    shape = item.get("embedding_shape")
    ref_s3_key = item.get("reference_audio_s3_key")
    ref_local = item.get("reference_audio_local_path")

    if data_b64 is not None:
        try:
            raw = base64.b64decode(data_b64)
            arr = np.frombuffer(raw, dtype=np.float32)
            if shape:
                arr = arr.reshape(tuple(int(x) for x in shape))
            session["reference_target_se"] = torch.from_numpy(arr.copy())
            if tv_id:
                session["trained_voice_id"] = tv_id
            if ref_s3_key:
                session["reference_audio_s3_key"] = ref_s3_key
            else:
                session.pop("reference_audio_s3_key", None)
            if ref_local:
                session["reference_audio_local_path"] = ref_local
            else:
                session.pop("reference_audio_local_path", None)
            return
        except (TypeError, ValueError, Exception) as e:
            logger.warning("Invalid embedding_b64 for sub: %s", e)
            return

    embedding_vals = item.get("embedding")
    if not embedding_vals:
        return
    try:
        flat = [float(v) for v in embedding_vals]
        session["reference_target_se"] = torch.tensor(flat)
        if tv_id:
            session["trained_voice_id"] = tv_id
        if ref_s3_key:
            session["reference_audio_s3_key"] = ref_s3_key
        else:
            session.pop("reference_audio_s3_key", None)
        if ref_local:
            session["reference_audio_local_path"] = ref_local
        else:
            session.pop("reference_audio_local_path", None)
    except (TypeError, ValueError) as e:
        logger.warning("Invalid embedding format: %s", e)


def _load_embedding_for_sub_into_session(sub: str, session: SessionData) -> None:
    """
    Load speaker embedding for a user from DynamoDB or local memory into the in-memory session.
    """
    logger.debug("Loading embedding for sub=%s", sub)
    if embeddings_table is not None:
        try:
            resp = embeddings_table.get_item(Key={"user_id": sub})
            logger.info("Loaded embedding for sub=%s: %s", sub, resp)
        except (BotoCoreError, ClientError) as e:
            logger.warning("Failed to load embedding for sub=%s: %s", sub, e)
            return
        item = resp.get("Item")
        if not item:
            return
        _apply_embedding_item_to_session(session, item)
        return

    item = _local_embeddings.get(sub)
    if item:
        _apply_embedding_item_to_session(session, item)


def _materialize_reference_to_temp(session: SessionData) -> Optional[str]:
    """
    Produce a temp WAV path for clone input. Caller must safe_remove the returned path.
    Copies local canonical WAV or downloads from S3.
    """
    local = session.get("reference_audio_local_path")
    if local and isinstance(local, str) and os.path.isfile(local):
        try:
            fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="ref_clone_")
            os.close(fd)
            shutil.copy2(local, tmp)
            return tmp
        except OSError as e:
            logger.warning("Failed to copy local reference: %s", e)
            return None
    s3k = session.get("reference_audio_s3_key")
    if s3k:
        return _download_reference_from_s3(s3k)
    return None


def _download_reference_from_s3(s3_key: str) -> Optional[str]:
    """Download reference audio from S3 to a temp file. Returns local path or None on failure."""
    if not s3_client or not S3_BUCKET:
        return None
    try:
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="ref_s3_")
        os.close(fd)
        s3_client.download_file(S3_BUCKET, s3_key, path)
        logger.debug("Downloaded reference from s3://%s/%s to %s", S3_BUCKET, s3_key, path)
        return path
    except (BotoCoreError, ClientError, OSError) as e:
        logger.warning("Failed to download reference from S3 key=%s: %s", s3_key, e)
        return None


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
        ref_s3_key, ref_local = _store_embedding_for_sub(
            sub, voice_id, target_se, reference_path=reference_path
        )
        if ref_s3_key:
            session["reference_audio_s3_key"] = ref_s3_key
        else:
            session.pop("reference_audio_s3_key", None)
        if ref_local:
            session["reference_audio_local_path"] = ref_local
        else:
            session.pop("reference_audio_local_path", None)
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
async def get_state(
    sub: Optional[str] = None,
    current_user: Optional[AuthedUser] = Depends(get_current_user_optional),
) -> StateResponse:
    # With ?sub=...: no auth; return state for that sub from sub_voice_links, 401 if expired/missing
    if sub is not None:
        sub = sub.strip()
        if not sub:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Link expired or invalid",
            )
        link = sub_voice_links.get(sub)
        if not link or time.time() > link["expires_at"]:
            if sub in sub_voice_links:
                sub_voice_links.pop(sub, None)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Link expired or invalid",
            )
        return StateResponse(
            session_id=sub,
            trained_voice_id=link["trained_voice_id"],
        )

    # No query param: bearer token required; updates expiry (registers link on first /state)
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Bearer token",
        )
    logger.info("GET /state sub=%s", current_user.sub)
    session = get_or_create_session(current_user.sub)
    # If we have a trained voice but no embedding in this process, attempt to hydrate from DynamoDB.
    trained_voice_id = session.get("trained_voice_id")
    if not trained_voice_id and not session.get("reference_target_se"):
        _load_embedding_for_sub_into_session(current_user.sub, session)
        trained_voice_id = session.get("trained_voice_id")
    # Register sub for /clone-sub on first /state; expires 1h after this first call
    if trained_voice_id and current_user.sub not in sub_voice_links:
        target_se = session.get("reference_target_se")
        target_audio_name = session.get("reference_audio_name")
        ref_s3_key = session.get("reference_audio_s3_key")
        ref_local = session.get("reference_audio_local_path")
        if target_se is not None and (ref_s3_key or ref_local):
            sub_voice_links[current_user.sub] = {
                "trained_voice_id": trained_voice_id,
                "reference_target_se": target_se,
                "reference_audio_name": target_audio_name,
                "reference_audio_s3_key": ref_s3_key,
                "reference_audio_local_path": ref_local,
                "expires_at": time.time() + SUB_LINK_TTL_SEC,
            }
            logger.debug("sub_voice_links registered sub=%s expires_at=%s", current_user.sub, sub_voice_links[current_user.sub]["expires_at"])
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

    # If there is no embedding in this process but we've persisted one, hydrate from DynamoDB.
    if target_se is None:
        _load_embedding_for_sub_into_session(current_user.sub, session)
        trained_voice_id = session.get("trained_voice_id")
        target_se = session.get("reference_target_se")

    if not trained_voice_id or target_se is None:
        logger.warning("clone rejected: no trained voice sub=%s", current_user.sub)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No trained voice registered for this session",
        )

    ref_path = _materialize_reference_to_temp(session)
    if not ref_path:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Reference audio not available. Complete training so reference audio is persisted.",
        )

    # Base audio is also truncated to 20 seconds if longer
    base_path = write_b64_audio_to_temp_wav(
        prefix=f"base_{current_user.sub}", audio_b64=body.base
    )
    try:
        output_path = run_clone(
            reference_speaker=ref_path,
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
        safe_remove(ref_path)

    logger.info("clone done sub=%s trained_voice_id=%s", current_user.sub, trained_voice_id)
    return CloneResponse(
        session_id=current_user.sub,
        trained_voice_id=trained_voice_id,
        output_path=output_path,
    )


def _clone_for_sub(sub: str, base_b64: str) -> CloneResponse:
    """Resolve sub → voice link (expires 1h after first /state), run clone, return response."""
    link = sub_voice_links.get(sub)
    if not link:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No voice link for this sub or link expired",
        )
    if time.time() > link["expires_at"]:
        sub_voice_links.pop(sub, None)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Voice link expired",
        )
    trained_voice_id = link["trained_voice_id"]
    target_se = link["reference_target_se"]
    target_audio_name = link.get("reference_audio_name")
    link_session: SessionData = {
        "reference_audio_s3_key": link.get("reference_audio_s3_key"),
        "reference_audio_local_path": link.get("reference_audio_local_path"),
    }
    ref_path = _materialize_reference_to_temp(link_session)
    if not ref_path:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Reference audio not available for this link",
        )
    base_path = write_b64_audio_to_temp_wav(prefix=f"base_sub_{sub}", audio_b64=base_b64)
    try:
        output_path = run_clone(
            reference_speaker=ref_path,
            base_speaker=base_path,
            target_se=target_se,
            target_audio_name=target_audio_name,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("clone-sub failed sub=%s: %s", sub, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate cloned voice: {exc}",
        ) from exc
    finally:
        safe_remove(base_path)
        safe_remove(ref_path)
    return CloneResponse(
        session_id=sub,
        trained_voice_id=trained_voice_id,
        output_path=output_path,
    )


@app.post("/clone-sub", response_model=CloneResponse)
async def clone_voice_sub(body: CloneSubRequest) -> CloneResponse:
    """Clone using sub to resolve voice_id; sub link expires 1h after first /state."""
    logger.info("POST /clone-sub sub=%s", body.sub)
    return _clone_for_sub(body.sub, body.base)


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
                # If this process doesn't yet have the embedding, try hydrating from DynamoDB.
                if target_se is None:
                    _load_embedding_for_sub_into_session(user.sub, session)
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
                ref_path = _materialize_reference_to_temp(session)
                if not ref_path:
                    await websocket.send_json(
                        {
                            "error": "Reference audio not available. Complete training so reference audio is persisted.",
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
                        reference_speaker=ref_path,
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
                    safe_remove(ref_path)

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


@app.websocket("/clone-sub")
async def clone_voice_sub_ws(websocket: WebSocket) -> None:
    """Clone using sub to resolve voice_id (no auth); sub link expires 1h after first /state."""
    await websocket.accept()
    logger.info("WS /clone-sub connected")
    try:
        while True:
            message = await websocket.receive_text()
            try:
                payload = json.loads(message)
                sub = payload.get("sub")
                base_b64 = payload.get("base")
                if not sub or not base_b64:
                    await websocket.send_json(
                        {"error": "Missing 'sub' or 'base' field", "code": 422}
                    )
                    continue
                result = _clone_for_sub(sub, base_b64)
                await websocket.send_json(
                    {
                        "type": "clone_result",
                        "session_id": result.session_id,
                        "trained_voice_id": result.trained_voice_id,
                        "output_path": result.output_path,
                    }
                )
            except HTTPException as e:
                await websocket.send_json(
                    {"error": e.detail or "Clone failed", "code": e.status_code}
                )
            except json.JSONDecodeError as e:
                logger.debug("WS /clone-sub invalid JSON: %s", e)
                await websocket.send_json({"error": "Invalid JSON", "code": 400})
    except WebSocketDisconnect:
        logger.info("WS /clone-sub disconnected")


if __name__ == "__main__":
    import uvicorn

    logger.info("starting uvicorn")
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)

