"""Password hashing with bcrypt.

Security properties:
- Passwords are NEVER stored or logged in plaintext. Only the bcrypt hash is.
- Verification is constant-time (bcrypt.checkpw) and never raises on malformed
  hashes - invalid input simply fails verification.
- Each hash embeds a random salt and the configured work factor, so identical
  passwords produce different hashes.
- needs_rehash() supports silent upgrade of legacy-cost hashes at login.
"""

from __future__ import annotations

import bcrypt

from backend.app.core.config import settings

# bcrypt only uses the first 72 bytes of a password. Rather than silently
# truncating, reject absurdly long inputs (also caps DoS via huge inputs).
MAX_PASSWORD_BYTES = 72
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128


class PasswordTooLongError(ValueError):
    """Raised when a password exceeds bcrypt's usable length."""


def _validate_plaintext(password: str) -> bytes:
    if not isinstance(password, str):
        raise ValueError("password must be a string")
    encoded = password.encode("utf-8")
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise PasswordTooLongError(
            f"password must be at most {MAX_PASSWORD_BYTES} bytes long"
        )
    return encoded


def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt. Never store or log the input."""
    encoded = _validate_plaintext(password)
    hashed = bcrypt.hashpw(encoded, bcrypt.gensalt(rounds=settings.auth_bcrypt_rounds))
    return hashed.decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time password verification. Never raises on malformed input."""
    try:
        encoded = _validate_plaintext(password)
    except (ValueError, PasswordTooLongError):
        return False
    if not isinstance(password_hash, str) or not password_hash:
        return False
    try:
        expected = password_hash.encode("ascii")
        return bcrypt.checkpw(encoded, expected)
    except (ValueError, UnicodeEncodeError):
        # Malformed/corrupt hash in the store: treat as failed verification
        # without leaking details to the caller.
        return False


def needs_rehash(password_hash: str) -> bool:
    """True when the stored hash's cost is below the configured work factor."""
    if not isinstance(password_hash, str) or not password_hash.startswith("$2"):
        return False
    try:
        rounds = int(password_hash.split("$")[2])
    except (IndexError, ValueError):
        return False
    return rounds < settings.auth_bcrypt_rounds
