import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app import auth


def test_login_and_token_roundtrip():
    assert auth.authenticate("analyst1", "wrong") is None
    assert auth.authenticate("nobody", "x") is None
    user = auth.authenticate("analyst1", "analyst123")
    token = auth.create_token(user)
    assert auth.current_user(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)) == user


def test_tampered_token_rejected():
    token = auth.create_token(auth.authenticate("viewer1", "viewer123"))
    with pytest.raises(HTTPException) as err:
        auth.current_user(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token[:-2] + "xx"))
    assert err.value.status_code == 401


def test_role_permissions():
    viewer, analyst, admin = (auth.authenticate(u, p) for u, p in
                              [("viewer1", "viewer123"), ("analyst1", "analyst123"), ("admin1", "admin123")])
    assert viewer.can_use("knowledge_search") and not viewer.can_use("search_incidents")
    assert analyst.can_use("python_analysis") and not analyst.can_use("create_incident")
    assert admin.can_use("create_incident") and admin.can_use("view_audit_log")
    assert "restricted" not in analyst.access_levels and "restricted" in admin.access_levels


def test_token_bucket_burst_then_refill(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(auth.time, "monotonic", lambda: clock[0])
    bucket = auth.TokenBucket({"viewer": (3, 0.5)})

    assert [bucket.take("u", "viewer") for _ in range(3)] == [0, 0, 0]
    wait = bucket.take("u", "viewer")
    assert wait == pytest.approx(2.0)  # one token at 0.5/s
    assert bucket.take("other", "viewer") == 0  # buckets are per user

    clock[0] += 2.0
    assert bucket.take("u", "viewer") == 0
