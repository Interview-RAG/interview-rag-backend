"""Router behaviour with fake deployments. No network."""
import asyncio
import json
import os
import sys
import time
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

import llm
import models_config
from llm_budget import Caps, Deployment


class FakeStatusError(Exception):
    def __init__(self, status_code, message="", headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = None
        self.response = types.SimpleNamespace(headers=headers or {})


def fake_response(content="ok", total_tokens=42, tool_calls=None):
    message = types.SimpleNamespace(content=content, tool_calls=tool_calls)
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=message)],
        usage=types.SimpleNamespace(total_tokens=total_tokens),
    )


@pytest.fixture
def two_deps(monkeypatch):
    a = Deployment("a", "pa", "model-a", "http://a", "k", rpm=10, tpm=10_000, rpd=100, caps=Caps())
    b = Deployment("b", "pb", "model-b", "http://b", "k", rpm=10, tpm=10_000, rpd=100, caps=Caps())
    monkeypatch.setattr(llm, "_DEPLOYMENTS", {"a": a, "b": b})
    monkeypatch.setattr(models_config, "TIERS", {"t": ["a", "b"]})
    monkeypatch.setattr(models_config, "LLM_MAX_WAIT_S", 0.5)
    return a, b


def run(coro):
    return asyncio.run(coro)


def test_first_deployment_serves_when_healthy(two_deps):
    a, b = two_deps
    calls = []

    async def fn(dep):
        calls.append(dep.name)
        return fake_response(total_tokens=77)

    out = run(llm.run("t", 100, fn, lambda r: r.usage.total_tokens))
    assert calls == ["a"] and out.usage.total_tokens == 77
    assert a.served == 1
    assert a.tpm.used(time.monotonic()) == 77  # settled to real usage, not the estimate
    assert a.tpm.used(time.monotonic() + 1e6) == 0  # far future: window drained


def test_429_cools_first_and_falls_through(two_deps):
    a, b = two_deps
    calls = []

    async def fn(dep):
        calls.append(dep.name)
        if dep.name == "a":
            raise FakeStatusError(429, "Rate limit reached. Please try again in 12.5s")
        return fake_response()

    run(llm.run("t", 100, fn))
    assert calls == ["a", "b"]
    assert a.failed == 1 and a.cooldown_reason == "rate_limited" and a.cooldown_until > 0
    assert b.served == 1


def test_daily_quota_message_cools_until_tomorrow(two_deps):
    a, b = two_deps

    async def fn(dep):
        if dep.name == "a":
            raise FakeStatusError(429, "Quota exceeded: GenerateRequestsPerDayPerProjectPerModel")
        return fake_response()

    run(llm.run("t", 100, fn))
    assert a.cooldown_reason == "daily_quota"
    assert a.cooldown_until - time.monotonic() > 60


def test_402_and_5xx_are_transient(two_deps):
    a, b = two_deps

    async def fn(dep):
        if dep.name == "a":
            raise FakeStatusError(402, "insufficient credits")
        return fake_response()

    run(llm.run("t", 100, fn))
    assert a.cooldown_reason == "no_credits"

    a.cooldown_until = 0

    async def fn2(dep):
        if dep.name == "a":
            raise FakeStatusError(503, "over capacity")
        return fake_response()

    run(llm.run("t", 100, fn2))
    assert a.cooldown_reason == "http_503"


def test_fatal_400_propagates_immediately(two_deps):
    a, b = two_deps

    async def fn(dep):
        raise FakeStatusError(400, "messages[0].content is required")

    with pytest.raises(FakeStatusError):
        run(llm.run("t", 100, fn))
    assert b.served == 0 and b.failed == 0  # never reached


def test_incompatible_400_is_skipped_for_an_hour(two_deps):
    a, b = two_deps

    async def fn(dep):
        if dep.name == "a":
            raise FakeStatusError(400, "reasoning_effort: value 'none' is not supported")
        return fake_response()

    run(llm.run("t", 100, fn))
    assert a.cooldown_reason == "incompatible" and b.served == 1


def test_410_retired_model_is_incompatible_and_413_has_no_cooldown(two_deps):
    a, b = two_deps

    async def fn(dep):
        if dep.name == "a":
            raise FakeStatusError(410, "The model has reached its end of life")
        return fake_response()

    run(llm.run("t", 100, fn))
    assert a.cooldown_reason == "incompatible" and b.served == 1

    a.cooldown_until = 0

    async def fn2(dep):
        if dep.name == "a":
            raise FakeStatusError(413, "Request too large for model")
        return fake_response()

    run(llm.run("t", 100, fn2))
    assert a.cooldown_until == 0 and a.failed == 2 and b.served == 2


def test_estimate_may_depend_on_deployment(two_deps):
    a, b = two_deps
    a.tpm.limit = 1000
    seen = {}

    async def fn(dep):
        return fake_response()

    def est(dep):
        seen[dep.name] = 5000 if dep.name == "a" else 100
        return seen[dep.name]

    run(llm.run("t", est, fn))
    assert seen == {"a": 5000, "b": 100}
    assert a.skipped == 1 and b.served == 1  # 5000 never fits a's 1000 TPM


def test_empty_response_falls_through(two_deps):
    a, b = two_deps

    async def fn(dep):
        return fake_response(content="   ") if dep.name == "a" else fake_response("real")

    out = run(llm.run("t", 100, fn))
    assert out.choices[0].message.content == "real" and a.cooldown_reason == "empty"


def test_budget_exhausted_skips_without_calling(two_deps):
    a, b = two_deps
    a.rpm.limit = 1
    calls = []

    async def fn(dep):
        calls.append(dep.name)
        return fake_response()

    run(llm.run("t", 10, fn))
    run(llm.run("t", 10, fn))
    assert calls == ["a", "b"] and a.skipped == 1


def test_all_unavailable_raises_llm_unavailable(two_deps):
    a, b = two_deps

    async def fn(dep):
        raise FakeStatusError(429, "slow down")  # 15s cooldown > LLM_MAX_WAIT_S → no sleep

    with pytest.raises(llm.LLMUnavailable) as e:
        run(llm.run("t", 100, fn))
    assert "a=rate_limited" in str(e.value) and "b=rate_limited" in str(e.value)


def test_waits_for_soonest_window_when_short(two_deps):
    a, b = two_deps
    b.disabled = True
    a.rpm.limit = 1
    a.rpm.seconds = 0.2  # tiny window so the test is quick
    calls = []

    async def fn(dep):
        calls.append(dep.name)
        return fake_response()

    run(llm.run("t", 10, fn))
    run(llm.run("t", 10, fn))  # first sweep skips, waits ~0.2s, second sweep serves
    assert calls == ["a", "a"]


def test_vision_and_tools_filters(two_deps):
    a, b = two_deps
    b.caps.vision = True

    async def fn(dep):
        return fake_response(dep.name)

    out = run(llm.run("t", 10, fn, needs_vision=True))
    assert out.choices[0].message.content == "b"
    a.caps.tools = False
    out = run(llm.run("t", 10, fn, needs_tools=True))
    assert out.choices[0].message.content == "b"
    b.caps.vision = False
    with pytest.raises(llm.LLMUnavailable):
        run(llm.run("t", 10, fn, needs_vision=True))


# --- normalisation ------------------------------------------------------------

def dep_with(**caps):
    return Deployment("x", "gemini", "m", "http://x", "k", rpm=1, tpm=1, rpd=1, caps=Caps(**caps))


def test_normalise_drops_or_aliases_reasoning_none():
    p = llm._normalise(dep_with(reasoning_none=False), {"messages": [], "reasoning_effort": "none"})
    assert "reasoning_effort" not in p
    p = llm._normalise(dep_with(reasoning_none=False, reasoning_none_alias="low"),
                       {"messages": [], "reasoning_effort": "none"})
    assert p["reasoning_effort"] == "low"
    p = llm._normalise(dep_with(reasoning_none=True), {"messages": [], "reasoning_effort": "none"})
    assert p["reasoning_effort"] == "none"
    p = llm._normalise(dep_with(reasoning_none=False, thinking_template_kwarg=True),
                       {"messages": [], "reasoning_effort": "none"})
    assert "reasoning_effort" not in p
    assert p["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_normalise_json_mode_fallback_nudges_last_user_message():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "give json"}]
    p = llm._normalise(dep_with(json_mode=False), {"messages": msgs, "response_format": {"type": "json_object"}})
    assert "response_format" not in p
    assert p["messages"][1]["content"].startswith("give json") and "JSON" in p["messages"][1]["content"]
    assert msgs[1]["content"] == "give json"  # caller's list untouched


def test_normalise_groq_drops_json_mode_when_tools_present():
    d = Deployment("g", "groq", "m", "http://g", "k", rpm=1, tpm=1, rpd=1, caps=Caps(json_mode=True))
    p = llm._normalise(d, {"messages": [{"role": "user", "content": "x"}],
                           "tools": [{"type": "function"}], "response_format": {"type": "json_object"}})
    assert "response_format" not in p


def test_normalise_clamps_max_tokens_and_fixes_messages():
    msgs = [
        {"role": "system", "content": "first"},
        {"role": "assistant", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "name": "web_search", "content": "r"},
        {"role": "system", "content": "stop searching"},
    ]
    p = llm._normalise(dep_with(system_first_only=True, max_output=100), {"messages": msgs, "max_tokens": 5000})
    assert p["max_tokens"] == 100
    assert p["messages"][1]["content"] == ""
    assert "name" not in p["messages"][2]
    assert p["messages"][0]["role"] == "system" and p["messages"][3]["role"] == "user"


def test_normalise_rejects_tools_on_no_tool_deployment():
    with pytest.raises(ValueError):
        llm._normalise(dep_with(tools=False), {"messages": [], "tools": [{}]})


# --- parse_json / strip_think ---------------------------------------------------

def test_strip_think_closed_and_truncated():
    assert llm.strip_think("<think>hmm</think>{\"a\":1}") == '{"a":1}'
    assert llm.strip_think("prefix <think>never closed") == "prefix"
    assert llm.strip_think(None) == ""


def test_parse_json_variants():
    assert llm.parse_json('{"a": 1}') == {"a": 1}
    assert llm.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.parse_json('<think>x</think>Sure! {"a": 1} hope that helps') == {"a": 1}
    assert llm.parse_json('here: ["React", "Python"]') == ["React", "Python"]
    assert llm.parse_json('{"outer": {"inner": [1, 2]}} trailing') == {"outer": {"inner": [1, 2]}}
    with pytest.raises(json.JSONDecodeError):
        llm.parse_json("no json here")
    with pytest.raises(json.JSONDecodeError):
        llm.parse_json("")


def test_chunk_chars_follows_first_usable_deployment(two_deps):
    a, b = two_deps
    a.caps.max_input_chars = 60_000
    b.caps.max_input_chars = 15_000
    assert llm.chunk_chars("t") == 60_000
    a.rpm.limit = 0
    a.rpm.add(0, 1)
    assert llm.chunk_chars("t") == 15_000
    b.disabled = True
    assert llm.chunk_chars("t") == 15_000  # min of the tier when nothing is usable
