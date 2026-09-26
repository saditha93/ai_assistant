"""Users, roles, permissions, login tokens and per-user rate limiting.

ROLE_POLICY is the single place that says what a role may see and do. Every
enforcement point (tool binding, tool execution, retrieval filters, API endpoints)
reads from it, so there is nothing to keep in sync.
"""

import datetime as dt
import hashlib
import hmac
import time
from dataclasses import dataclass

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

ACCESS_LEVELS = ["public", "internal", "confidential", "restricted"]

SEARCH_TOOLS = {"knowledge_search"}
ANALYST_TOOLS = {"deep_research", "python_analysis", "search_employees", "get_service", "search_incidents"}
ADMIN_TOOLS = {"create_incident", "view_audit_log"}
WRITE_TOOLS = {"create_incident"}  # need human approval before they run

ROLE_POLICY = {
    "viewer": {"access_levels": ACCESS_LEVELS[:2], "tools": SEARCH_TOOLS},
    "analyst": {"access_levels": ACCESS_LEVELS[:3], "tools": SEARCH_TOOLS | ANALYST_TOOLS},
    "admin": {"access_levels": ACCESS_LEVELS, "tools": SEARCH_TOOLS | ANALYST_TOOLS | ADMIN_TOOLS},
}


@dataclass(frozen=True)
class User:
    username: str
    name: str
    role: str
    department: str

    @property
    def tools(self) -> set[str]:
        return ROLE_POLICY[self.role]["tools"]

    @property
    def access_levels(self) -> list[str]:
        return ROLE_POLICY[self.role]["access_levels"]

    def can_use(self, tool: str) -> bool:
        return tool in self.tools


# Option A from the brief: hardcoded users. Passwords are stored as salted PBKDF2 hashes.
# Demo passwords: viewer123 / analyst123 / admin123
USERS = {
    "viewer1": {
        "name": "Kasun Silva", "role": "viewer", "department": "retail",
        "password": "e1bf76f827ae60dd$2a4c1cf6ac53c4194c215b5b0760c2632a138a6a8f7e33072c13acdb6f8620b6",
    },
    "analyst1": {
        "name": "Dilani Fernando", "role": "analyst", "department": "payments",
        "password": "f801eb84708129ea$6777a2453cb2d5100001da6137d7e836fe12c3d503f8b058ef1564baec42b548",
    },
    "admin1": {
        "name": "Ruwan Jayasinghe", "role": "admin", "department": "platform",
        "password": "20733cf44b9cd399$3c3a4295b47e43d60080d310285bb88c87fcead5a8c1958d9c422995d7d47880",
    },
}


def _user(username: str) -> User:
    record = USERS[username]
    return User(username, record["name"], record["role"], record["department"])


def authenticate(username: str, password: str) -> User | None:
    record = USERS.get(username)
    if not record:
        return None
    salt, expected = record["password"].split("$")
    actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 100_000).hex()
    return _user(username) if hmac.compare_digest(actual, expected) else None


def create_token(user: User) -> str:
    expires = dt.datetime.now(dt.UTC) + dt.timedelta(minutes=settings.jwt_ttl_minutes)
    return jwt.encode({"sub": user.username, "exp": expires}, settings.jwt_secret, algorithm="HS256")


_bearer = HTTPBearer(auto_error=False)


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> User:
    """The token only proves who you are. The role is looked up here on every request,
    so it can never come from the client or from anything the LLM produced."""
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    try:
        claims = jwt.decode(creds.credentials, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token") from exc
    if claims.get("sub") not in USERS:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unknown user")
    return _user(claims["sub"])


def require_admin(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Administrator role required")
    return user


class TokenBucket:
    """Classic token bucket, one bucket per user.

    Each request takes one token. Tokens refill continuously at `refill` per second up to
    `capacity`, so users can burst briefly but not sustain more than the refill rate.
    No lock is needed: take() never awaits, so on one event loop it runs atomically.
    """

    def __init__(self, limits: dict[str, tuple[int, float]]):
        self.limits = limits
        self.buckets: dict[str, list[float]] = {}  # username -> [tokens, last_refill_time]

    def take(self, username: str, role: str) -> float:
        """Return 0 if allowed, otherwise the seconds until a token is available."""
        capacity, refill = self.limits[role]
        now = time.monotonic()
        tokens, last = self.buckets.get(username, [capacity, now])
        tokens = min(capacity, tokens + (now - last) * refill)
        if tokens >= 1:
            self.buckets[username] = [tokens - 1, now]
            return 0.0
        self.buckets[username] = [tokens, now]
        return (1 - tokens) / refill


# ponytail: in-memory buckets, one API process. Move to Redis if we run several replicas.
rate_limiter = TokenBucket(settings.rate_limits)


def rate_limited_user(user: User = Depends(current_user)) -> User:
    wait = rate_limiter.take(user.username, user.role)
    if wait:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Rate limit reached for role '{user.role}'. Please try again in {wait:.0f} seconds.",
            headers={"Retry-After": str(max(1, round(wait)))},
        )
    return user
