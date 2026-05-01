import base64
from typing import Any, Dict

import sys

import pytest
from fastapi.testclient import TestClient

import main
from models import AuthedUser


@pytest.fixture(autouse=True)
def clear_sessions() -> None:
    """
    Ensure in-memory session store is empty before each test.
    """
    main.sessions.clear()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """
    TestClient with auth and cloning behavior patched for tests.
    """

    # Dummy authenticated user
    user = AuthedUser(sub="test-user", email="test@example.com", raw_claims={})

    async def fake_get_current_user(*args: Any, **kwargs: Dict[str, Any]) -> AuthedUser:  # type: ignore[override]
        return user

    def fake_verify_bearer_or_jwt(token: str) -> AuthedUser:
        return user

    def fake_extract_token_from_websocket(websocket) -> str:
        # Token value is irrelevant since verify_bearer_or_jwt is patched
        return "dummy-token"

    def fake_run_clone(*args: Any, **kwargs: Dict[str, Any]) -> str:
        # Pretend cloning succeeded and returned base64 PCM (backend returns PCM now)
        return base64.b64encode(b"fake-pcm-output").decode("utf-8")

    def fake_extract_speaker_embedding(audio_path: str) -> tuple[Any, str]:
        return (object(), "test_audio")

    # Patch auth / cloning entrypoints used by main.py
    monkeypatch.setattr(main, "get_current_user", fake_get_current_user)
    monkeypatch.setattr(main, "verify_bearer_or_jwt", fake_verify_bearer_or_jwt)
    monkeypatch.setattr(main, "extract_token_from_websocket", fake_extract_token_from_websocket)
    monkeypatch.setattr(main, "run_clone", fake_run_clone)
    monkeypatch.setattr(main, "extract_speaker_embedding", fake_extract_speaker_embedding)

    return TestClient(main.app)


def make_b64_audio(payload: bytes | None = None) -> str:
    """
    Helper to make a small base64 "audio" payload for tests.
    """
    if payload is None:
        payload = b"fake-wav-audio"
    return base64.b64encode(payload).decode("utf-8")


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_verify_bearer_or_jwt_admin_matches_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    voice_auth = sys.modules["auth"]
    monkeypatch.setattr(voice_auth, "BEARER_TOKEN", "admin-shared-secret")
    u = voice_auth.verify_bearer_or_jwt("admin-shared-secret")
    assert u.sub == "admin"
    assert u.email == "admin@local"
    assert u.raw_claims.get("admin") is True


def test_state_initial_has_no_trained_voice(client: TestClient) -> None:
    resp = client.get("/state", headers={"Authorization": "Bearer dummy"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "test-user"
    assert data["trained_voice_id"] is None


def test_train_then_state_and_clone_rest(client: TestClient) -> None:
    reference_b64 = make_b64_audio()
    start_resp = client.post(
        "/train",
        headers={"Authorization": "Bearer dummy"},
        json={"operation": "start"},
    )
    assert start_resp.status_code == 200
    assert start_resp.json() == {"status": "started"}

    chunk_resp = client.post(
        "/train",
        headers={"Authorization": "Bearer dummy"},
        json={"reference": reference_b64},
    )
    assert chunk_resp.status_code == 200
    assert chunk_resp.json() == {"status": "chunk_received"}

    end_resp = client.post(
        "/train",
        headers={"Authorization": "Bearer dummy"},
        json={"operation": "end"},
    )
    assert end_resp.status_code == 200
    train_data = end_resp.json()
    assert train_data["session_id"] == "test-user"
    assert isinstance(train_data["trained_voice_id"], str)
    assert len(train_data["trained_voice_id"]) == 6

    state_resp = client.get("/state", headers={"Authorization": "Bearer dummy"})
    assert state_resp.status_code == 200
    state_data = state_resp.json()
    assert state_data["trained_voice_id"] == train_data["trained_voice_id"]

    base_b64 = make_b64_audio(b"base-audio")
    clone_resp = client.post(
        "/clone",
        headers={"Authorization": "Bearer dummy"},
        json={"base": base_b64},
    )
    assert clone_resp.status_code == 200
    clone_data = clone_resp.json()
    assert clone_data["session_id"] == "test-user"
    assert clone_data["trained_voice_id"] == train_data["trained_voice_id"]
    assert "output_path" in clone_data


def test_train_reference_without_start_returns_422(client: TestClient) -> None:
    resp = client.post(
        "/train",
        headers={"Authorization": "Bearer dummy"},
        json={"reference": make_b64_audio()},
    )
    assert resp.status_code == 422
    assert "start" in (resp.json().get("detail") or "").lower()


def test_clone_without_training_rest_returns_422(client: TestClient) -> None:
    base_b64 = make_b64_audio()
    resp = client.post(
        "/clone",
        headers={"Authorization": "Bearer dummy"},
        json={"base": base_b64},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"] == "No trained voice registered for this session"


def test_train_and_clone_via_websocket(client: TestClient) -> None:
    reference_b64 = make_b64_audio()
    base_b64 = make_b64_audio(b"base-audio")

    with client.websocket_connect("/train?token=dummy") as ws:
        ws.send_json({"operation": "start"})
        msg = ws.receive_json()
        assert msg["type"] == "started"

        ws.send_json({"reference": reference_b64})
        msg = ws.receive_json()
        assert msg["type"] == "chunk_received"

        ws.send_json({"operation": "end"})
        msg = ws.receive_json()
        assert msg["type"] == "train_result"
        assert msg["session_id"] == "test-user"
        assert isinstance(msg["trained_voice_id"], str)
        assert len(msg["trained_voice_id"]) == 6

    with client.websocket_connect("/clone?token=dummy") as ws:
        ws.send_json({"base": base_b64})
        msg = ws.receive_json()
        assert msg["type"] == "clone_result"
        assert msg["session_id"] == "test-user"
        assert isinstance(msg["trained_voice_id"], str)
        assert "output_path" in msg

