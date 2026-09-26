import time

from app import llm
from app.config import settings

DAILY = "429 RESOURCE_EXHAUSTED quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'"
MINUTE = "429 RESOURCE_EXHAUSTED ... Please retry in 7.5s. quotaId': 'GenerateRequestsPerMinute'"


def test_quota_errors_skip_models(monkeypatch):
    monkeypatch.setattr(llm, "EXHAUSTED", {})
    monkeypatch.setattr(settings, "gemini_model", "flash-a, flash-b")
    monkeypatch.setattr(settings, "gemini_fallback_model", "lite-a")

    llm.record_quota_error("flash-a", DAILY)
    llm.record_quota_error("lite-a", MINUTE)
    llm.record_quota_error("flash-b", "500 internal error")  # not a quota error

    assert llm.EXHAUSTED["flash-a"] > time.time() + 60
    assert 5 < llm.EXHAUSTED["lite-a"] - time.time() < 10
    assert llm.model_chain() == ["flash-b"]
    assert llm.model_chain(light=True) == ["flash-b"]


def test_all_models_exhausted_means_no_llm(monkeypatch):
    monkeypatch.setattr(llm, "EXHAUSTED", {})
    monkeypatch.setattr(settings, "google_api_key", "key")
    monkeypatch.setattr(settings, "gemini_model", "flash-a")
    monkeypatch.setattr(settings, "gemini_fallback_model", "lite-a")
    assert llm.llm_available()
    for m in ("flash-a", "lite-a"):
        llm.record_quota_error(m, DAILY)
    assert not llm.llm_available()
