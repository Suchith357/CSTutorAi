"""Security primitives for OSTutorAI: password hashing and JWT tokens."""

from backend.app.security.passwords import (
    hash_password,
    needs_rehash,
    verify_password,
)
from backend.app.security.tokens import (
    TokenError,
    create_access_token,
    decode_access_token,
)

__all__ = [
    "TokenError",
    "create_access_token",
    "decode_access_token",
    "hash_password",
    "needs_rehash",
    "verify_password",
]
