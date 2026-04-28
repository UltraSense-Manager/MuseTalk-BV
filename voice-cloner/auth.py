import logging
import os
from typing import Any, Dict

import boto3
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
DDB_TABLE_NAME = os.getenv("DDB_TABLE_NAME", "")
AWS_REGION = os.getenv("AWS_REGION")

if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET environment variable must be set")

if not DDB_TABLE_NAME:
    raise RuntimeError("DDB_TABLE_NAME environment variable must be set")

if not AWS_REGION:
    raise RuntimeError("AWS_REGION environment variable must be set")


ddb = boto3.resource("dynamodb", region_name=AWS_REGION)
user_table = ddb.Table(DDB_TABLE_NAME)

security = HTTPBearer(auto_error=True)


def verify_jwt_and_load_user(token: str) -> AuthedUser:
    """
    Decode a JWT, verify expiry and match against DynamoDB.

    Assumptions:
    - JWT contains at least `sub` and `email` claims.
    - DynamoDB table has partition key `sub` and attribute `email`.
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

    if not sub or not email:
        logger.warning("verify_jwt_and_load_user: missing sub or email")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing required claims",
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


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> AuthedUser:
    token = credentials.credentials
    logger.debug("get_current_user: verifying Bearer token")
    return verify_jwt_and_load_user(token)


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

