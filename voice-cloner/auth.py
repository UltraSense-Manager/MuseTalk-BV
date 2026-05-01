import logging
import os
from typing import Any, Dict, Optional

import jwt
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import Depends, HTTPException, WebSocket, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import ExpiredSignatureError, InvalidTokenError

import dotenv
from models import AuthedUser

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
# Same admin override as MuseTalk `musetalk.service.api` (`BEARER_TOKEN` matches raw bearer value).
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "").strip()
DDB_TABLE_NAME = os.getenv("DDB_TABLE_NAME", "").strip()
AWS_REGION = os.getenv("AWS_REGION", "").strip()

if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET environment variable must be set")

# DynamoDB user verification is optional: when table/region are not set, trust JWT claims only.
USE_DDB_USER_TABLE = bool(DDB_TABLE_NAME and AWS_REGION)
user_table = None
if USE_DDB_USER_TABLE:
    import boto3

    try:
        ddb = boto3.resource("dynamodb", region_name=AWS_REGION)
        user_table = ddb.Table(DDB_TABLE_NAME)
        logger.info("Voice cloner auth: DynamoDB user table enabled table=%s", DDB_TABLE_NAME)
    except (BotoCoreError, ClientError, Exception) as e:  # noqa: BLE001
        logger.warning("Voice cloner auth: DynamoDB init failed (%s); falling back to JWT-only", e)
        user_table = None
        USE_DDB_USER_TABLE = False
else:
    logger.info("Voice cloner auth: JWT-only (DDB_TABLE_NAME / AWS_REGION not set)")

security = HTTPBearer(auto_error=True)
security_optional = HTTPBearer(auto_error=False)


def verify_jwt_and_load_user(token: str) -> AuthedUser:
    """
    Decode JWT and verify user.

    With DynamoDB: require sub + email in token and match DynamoDB row (user_id partition key).
    Without DynamoDB: require sub in token; email from token or placeholder (no DB lookup).
    """
    logger.debug("verify_jwt_and_load_user: decoding token")
    try:
        payload: Dict[str, Any] = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["exp"]},
        )
    except ExpiredSignatureError:
        logger.warning("verify_jwt_and_load_user: token expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except InvalidTokenError as e:
        logger.warning("verify_jwt_and_load_user: invalid token: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    sub = payload.get("sub")
    email = payload.get("email")

    if not sub:
        logger.warning("verify_jwt_and_load_user: missing sub")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing required claim: sub",
        )

    if not USE_DDB_USER_TABLE or user_table is None:
        if not email:
            email = f"{sub}@jwt-local"
        logger.debug("verify_jwt_and_load_user: jwt-only ok sub=%s", sub)
        return AuthedUser(sub=str(sub), email=str(email), raw_claims=payload)

    if not email:
        logger.warning("verify_jwt_and_load_user: missing email for DynamoDB mode")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing required claim: email",
        )

    logger.debug("verify_jwt_and_load_user: lookup DynamoDB sub=%s", sub)
    try:
        ddb_resp = user_table.get_item(Key={"user_id": sub})
    except (ClientError, BotoCoreError) as e:
        logger.exception("verify_jwt_and_load_user: DynamoDB error sub=%s: %s", sub, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to access user store",
        )

    item = ddb_resp.get("Item")
    if not item or item.get("email") != email:
        logger.warning("verify_jwt_and_load_user: user not found or email mismatch sub=%s", sub)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not authorized",
        )

    logger.debug("verify_jwt_and_load_user: ok sub=%s email=%s", sub, email)
    return AuthedUser(sub=sub, email=email, raw_claims=payload)


def verify_bearer_or_jwt(token: str) -> AuthedUser:
    """
    Accept the configured admin `BEARER_TOKEN` (same as MuseTalk service) or a user JWT.

    Admin path uses `sub` ``admin`` so sessions align with `/api/job` when using the admin token.
    """
    stripped = (token or "").strip()
    if BEARER_TOKEN and stripped == BEARER_TOKEN:
        logger.debug("verify_bearer_or_jwt: admin BEARER_TOKEN match")
        return AuthedUser(
            sub="admin",
            email="admin@local",
            raw_claims={"sub": "admin", "admin": True},
        )
    return verify_jwt_and_load_user(stripped)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> AuthedUser:
    token = credentials.credentials
    logger.debug("get_current_user: verifying Bearer token")
    return verify_bearer_or_jwt(token)


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_optional),
) -> Optional[AuthedUser]:
    """Returns AuthedUser if valid Bearer token present, else None."""
    if not credentials:
        return None
    try:
        return verify_bearer_or_jwt(credentials.credentials)
    except HTTPException:
        return None


def extract_token_from_websocket(websocket: WebSocket) -> str:
    """
    Extract Bearer token from WebSocket `Authorization` header or `token` query param.
    """
    logger.debug("extract_token_from_websocket")
    auth_header = websocket.headers.get("authorization") or websocket.headers.get(
        "Authorization"
    )
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.split(" ", 1)[1]

    token_param = websocket.query_params.get("token")
    if token_param:
        return token_param

    logger.warning("extract_token_from_websocket: no token in header or query")
    raise HTTPException(
        status_code=status.WS_1008_POLICY_VIOLATION,
        detail="Missing Bearer token",
    )
