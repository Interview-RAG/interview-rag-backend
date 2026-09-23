"""Budget maths with a fake clock. No network."""
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from llm_budget import (
    Caps,
    Deployment,
    Window,
    estimate_tokens,
    mentions_daily_quota,
    parse_retry_delay,
)


class FakeErr(Exception):
    def __init__(self, message="", headers=None, body=None):
        super().__init__(message)
        self.body = body
        if headers is not None:
            self.response = type("R", (), {"headers": headers})()


# --- Window -----------------------------------------------------------------

def test_window_fits_until_limit_then_reports_wait():
    w = Window(60, 3)
    assert w.wait_for(0, 1) == 0.0
    w.add(0, 1)
    w.add(10, 1)
    w.add(20, 1)
    assert w.used(20) == 3
    wait = w.wait_for(25, 1)
    # Oldest entry (t=0) expires at t=60 → wait 35s (plus a small pad).
    assert 35.0 <= wait < 35.2
    # Two more units need the entries at t=0 and t=10 to expire → t=70.
    wait2 = w.wait_for(25, 2)
    assert 45.0 <= wait2 < 45.2


def test_window_prunes_expired_entries():
    w = Window(60, 2)
    w.add(0, 1)
    w.add(1, 1)
    assert w.wait_for(30, 1) > 0
    assert w.wait_for(61.1, 1) == 0.0
    assert w.used(61.1) == 0


def test_window_never_fits_when_request_exceeds_limit():
    w = Window(60, 8000)
    assert w.wait_for(0, 9000) == math.inf


def test_window_unlimited():
    w = Window(60, None)
    for i in range(100):
        w.add(i, 10_000)
    assert w.wait_for(50, 10**9) == 0.0


def test_window_entry_can_be_corrected_after_the_fact():
    w = Window(60, 1000)
    entry = w.add(0, 900)          # estimate
    assert w.wait_for(1, 200) > 0  # would not fit
    entry[1] = 300                 # settle to real usage
    assert w.wait_for(1, 200) == 0.0


# --- Deployment -------------------------------------------------------------

def make_dep(**kw):
    defaults = dict(rpm=2, tpm=1000, rpd=5, caps=Caps())
    defaults.update(kw)
    return Deployment("d", "p", "m", "http://x", "key", **defaults)


def test_check_reserve_settle_roundtrip():
    d = make_dep()
    ok, wait, reason = d.check(0, 400)
    assert ok and wait == 0 and reason == ""
    res = d.reserve(0, 400)
    assert d.tpm.used(0) == 400 and d.rpm.used(0) == 1 and d.rpd.used(0) == 1
    d.settle(res, 150)
    assert d.tpm.used(0) == 150
    assert d.rpm.used(0) == 1  # request count is never rewritten


def test_check_reports_which_window_is_full():
    d = make_dep(rpm=1)
    d.reserve(0, 10)
    ok, wait, reason = d.check(1, 10)
    assert not ok and reason == "rpm" and 58 < wait < 60


def test_check_tpm_wait_and_oversize_request():
    d = make_dep(tpm=1000)
    d.reserve(0, 900)
    ok, wait, reason = d.check(1, 200)
    assert not ok and reason == "tpm" and wait > 0
    ok, wait, reason = d.check(1, 5000)
    assert not ok and reason == "tpm" and wait == math.inf


def test_daily_window_checked_before_minute_windows():
    d = make_dep(rpd=1)
    d.reserve(0, 10)
    ok, wait, reason = d.check(120, 10)   # minute windows have drained
    assert not ok and reason == "rpd" and wait > 3600


def test_shared_pool_is_charged_and_checked():
    pool = Window(86_400, 1000)
    a = make_dep(pools=[("tpd", pool)])
    b = make_dep(pools=[("tpd", pool)])
    a.reserve(0, 700)
    ok, wait, reason = b.check(1, 500)
    assert not ok and reason == "tpd"
    ok, _, _ = b.check(1, 200)
    assert ok


def test_cooldown_and_disabled_and_missing_key():
    d = make_dep()
    d.cool(0, 30, "rate_limited")
    ok, wait, reason = d.check(10, 1)
    assert not ok and reason == "cooldown:rate_limited" and wait == pytest.approx(20)
    d.cool(10, 5, "shorter")       # never shortens an existing cooldown
    assert d.cooldown_until == 30 and d.cooldown_reason == "rate_limited"
    ok, _, _ = d.check(31, 1)
    assert ok

    d.disabled = True
    assert d.check(31, 1) == (False, math.inf, "disabled")
    d.disabled = False
    d.api_key = None
    assert d.check(31, 1) == (False, math.inf, "no_key")
    assert not d.enabled()


def test_state_snapshot_shape():
    d = make_dep()
    d.reserve(0, 100)
    s = d.state(0)
    assert s["rpm"] == {"used": 1, "limit": 2}
    assert s["tpm"] == {"used": 100, "limit": 1000}
    assert s["cooldown_s"] == 0 and s["enabled"] is True


# --- estimate_tokens --------------------------------------------------------

def test_estimate_counts_text_images_tools_and_output():
    msgs = [
        {"role": "system", "content": "a" * 400},
        {"role": "user", "content": [
            {"type": "text", "text": "b" * 40},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]},
    ]
    base = estimate_tokens(msgs, max_tokens=None)
    # 440 chars / 4 = 110, + 2 msgs * 4, + 1000 image, + 1024 default output
    assert base == 110 + 8 + 1000 + 1024
    assert estimate_tokens(msgs, max_tokens=500) == 110 + 8 + 1000 + 500
    with_tools = estimate_tokens(msgs, 500, tools=[{"name": "x", "parameters": {}}])
    assert with_tools > estimate_tokens(msgs, 500)


def test_estimate_accepts_langchain_like_objects():
    class Msg:
        def __init__(self, content, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls
    msgs = [Msg("hello world"), Msg("", tool_calls=[{"name": "f", "args": {"q": "x" * 100}}])]
    assert estimate_tokens(msgs, 100) > 100


# --- parse_retry_delay / mentions_daily_quota ---------------------------------

def test_retry_after_header_seconds():
    assert parse_retry_delay(FakeErr(headers={"retry-after": "7"})) == 7.0


def test_retry_after_ms_header_wins():
    err = FakeErr(headers={"retry-after-ms": "2500", "retry-after": "7"})
    assert parse_retry_delay(err) == 2.5


def test_groq_try_again_in_body():
    assert parse_retry_delay(FakeErr("Rate limit reached. Please try again in 2.5s.")) == 3.0
    assert parse_retry_delay(FakeErr("Please try again in 1m2.5s")) == 63.0
    assert parse_retry_delay(FakeErr("Please try again in 250ms")) == pytest.approx(0.75)


def test_gemini_retry_delay_forms():
    assert parse_retry_delay(FakeErr("Please retry in 23.4s.")) == 23.9
    assert parse_retry_delay(FakeErr("quota", body={"error": {"details": [{"retryDelay": "23s"}]}})) == 23.5


def test_no_hint_returns_none():
    assert parse_retry_delay(FakeErr("boom")) is None


def test_daily_quota_detection():
    assert mentions_daily_quota(FakeErr("Quota exceeded for GenerateRequestsPerDayPerProjectPerModel"))
    assert mentions_daily_quota(FakeErr("Limit 200000, Used 199500, Requested 900 tokens per day"))
    assert not mentions_daily_quota(FakeErr("Rate limit reached for tokens per minute"))
