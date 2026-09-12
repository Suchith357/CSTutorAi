"""JWT bearer access tokens (HS256 family, signed with AUTH_SECRET_KEY).

Security properties:
- Tokens are signed (HMAC) and carry sub, exp, iat, jti, and a type claim.
- decode_access_token() verifies signature, expiry, and token type; any
  problem raises TokenError with a message that is safe to show clients.
- jti (unique token id) enables future revocation/denylisting without a
  format change.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from backend.app.core.config import settings

TOKEN_TYPE_ACCESS = "access"


class TokenError(Exception):
    """Raised when a token cannot be trusted (bad signature, expired, etc.)."""


def _secret() -> str:
    return settings.auth_secret_key


def create_access_token(subject: str) -> str:
    """Create a signed access token for a user id / username."""
    if not subject:
        raise ValueError("token subject must not be empty")
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=settings.auth_access_token_expire_minutes)
    payload: dict[str, Any] = {
        "sub": subject,
        "type": TOKEN_TYPE_ACCESS,
        "iat": now,
        "exp": expires,
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, _secret(), algorithm=settings.auth_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """Verify and decode an access token; raise TokenError if untrusted."""
    if not token or not isinstance(token, str):
        raise TokenError("Missing authentication token")
    try:
        payload = jwt.decode(
            token,
            _secret(),
            algorithms=[settings.auth_algorithm],
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("Token has expired; please log in again") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError("Invalid authentication token") from exc

    if payload.get("type") != TOKEN_TYPE_ACCESS:
        raise TokenError("Invalid token type")
    if not payload.get("sub"):
        raise TokenError("Token has no subject")
    return payload
