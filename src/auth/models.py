"""User authentication models and business logic."""
import hashlib
import hmac
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Optional


PASSWORD_MIN_LENGTH = 8
SESSION_TTL_SECONDS = 3600  # 1 hour
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_DURATION_SECONDS = 900  # 15 minutes

_sessions: dict[str, dict] = {}
_failed_attempts: dict[str, list[float]] = {}


@dataclass
class User:
    id: str
    email: str
    password_hash: str
    role: str = "user"
    is_active: bool = True
    created_at: float = field(default_factory=time.time)


class AuthError(Exception):
    pass


class AccountLockedError(AuthError):
    pass


class InvalidCredentialsError(AuthError):
    pass


def validate_email(email: str) -> bool:
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email))


def validate_password_strength(password: str) -> tuple[bool, list[str]]:
    """Returns (is_valid, list_of_failures)."""
    failures = []
    if len(password) < PASSWORD_MIN_LENGTH:
        failures.append(f"Password must be at least {PASSWORD_MIN_LENGTH} characters")
    if not re.search(r"[A-Z]", password):
        failures.append("Password must contain at least one uppercase letter")
    if not re.search(r"[a-z]", password):
        failures.append("Password must contain at least one lowercase letter")
    if not re.search(r"\d", password):
        failures.append("Password must contain at least one digit")
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        failures.append("Password must contain at least one special character")
    return len(failures) == 0, failures


def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hmac.new(
        salt.encode(), password.encode(), hashlib.sha256
    ).hexdigest()
    return hashed, salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    computed, _ = hash_password(password, salt)
    return hmac.compare_digest(computed, stored_hash)


def is_account_locked(user_id: str) -> bool:
    now = time.time()
    attempts = _failed_attempts.get(user_id, [])
    recent = [t for t in attempts if now - t < LOCKOUT_DURATION_SECONDS]
    _failed_attempts[user_id] = recent
    return len(recent) >= MAX_FAILED_ATTEMPTS


def record_failed_attempt(user_id: str) -> None:
    _failed_attempts.setdefault(user_id, []).append(time.time())


def create_session(user: User) -> str:
    token = secrets.token_urlsafe(32)
    _sessions[token] = {
        "user_id": user.id,
        "role": user.role,
        "created_at": time.time(),
        "expires_at": time.time() + SESSION_TTL_SECONDS,
    }
    return token


def get_session(token: str) -> Optional[dict]:
    session = _sessions.get(token)
    if session is None:
        return None
    if time.time() > session["expires_at"]:
        del _sessions[token]
        return None
    return session


def revoke_session(token: str) -> bool:
    if token in _sessions:
        del _sessions[token]
        return True
    return False


def require_role(session: dict, required_role: str) -> bool:
    role_hierarchy = {"admin": 3, "moderator": 2, "user": 1}
    user_level = role_hierarchy.get(session.get("role", ""), 0)
    required_level = role_hierarchy.get(required_role, 0)
    return user_level >= required_level
