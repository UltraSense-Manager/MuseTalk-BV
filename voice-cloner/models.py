from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, model_validator


class TrainRequest(BaseModel):
    operation: Optional[Literal["start", "end"]] = None  # "start" to begin chunking, "end" to finish and train
    reference: Optional[str] = None  # base64-encoded PCM chunk

    @model_validator(mode="after")
    def require_operation_or_reference(self) -> "TrainRequest":
        if self.operation is None and self.reference is None:
            raise ValueError("Either 'operation' or 'reference' must be set")
        return self


class TrainStartResponse(BaseModel):
    status: Literal["started"] = "started"


class TrainChunkResponse(BaseModel):
    status: Literal["chunk_received"] = "chunk_received"


class TrainResponse(BaseModel):
    session_id: str
    trained_voice_id: str


class CloneRequest(BaseModel):
    base: str  # base64-encoded audio to clone


class CloneResponse(BaseModel):
    session_id: str
    trained_voice_id: str
    output_path: str


class StateResponse(BaseModel):
    session_id: str
    trained_voice_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str


class AuthedUser(BaseModel):
    sub: str
    email: str
    raw_claims: Dict[str, Any]

