"""The one door every LLM call goes through.

    response = await llm.chat("heavy", messages=[...], temperature=0.2)

Picks a deployment for the tier from `models_config.TIERS`, checks its budget
(`llm_budget`), normalises the request for that provider, calls it, and falls
through to the next deployment on 429/5xx/402 honouring Retry-After. Returns
the OpenAI-shaped `ChatCompletion` so callers keep reading
`response.choices[0].message.content`, with any `<think>` block already gone.

`run()` is the same loop for callers that need to drive the client themselves
(the LangGraph agent, which goes through LangChain's `ChatOpenAI`).

All three providers speak the OpenAI chat-completions dialect, so one
`AsyncOpenAI` client per provider is enough. The SDK's own retries are turned
off: on a free tier a rate-limited request only clears once the minute window
drains, and sub-second retries just burn it faster.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

import models_config
from llm_budget import Caps, Deployment, Window, estimate_tokens, mentions_daily_quota, parse_retry_delay

logger = logging.getLogger("llm")

T = TypeVar("T")


class LLMUnavailable(Exception):
    """No deployment could serve the request: all busy, cooling down or unconfigured.

    Distinct from "the model ran and found nothing", which callers must be
    able to report differently.
    """


# --- Build deployments from config ------------------------------------------

def _scaled(limit: int | None) -> int | None:
    if limit is None:
        return None
    return max(1, int(limit * models_config.LLM_SAFETY))


def _build() -> dict[str, Deployment]:
    import os

    pools: dict[str, list[tuple[str, Window]]] = {}
    for pname, p in models_config.PROVIDERS.items():
        pools[pname] = []
        if p.get("tpd"):
            pools[pname].append(("tpd", Window(86_400, _scaled(p["tpd"]))))

    deployments: dict[str, Deployment] = {}
    for name, d in models_config.DEPLOYMENTS.items():
        provider = models_config.PROVIDERS[d["provider"]]
        env_name = name.upper().replace("-", "_")
        caps = Caps(
            json_mode=d.get("json_mode", True),
            reasoning_none=d.get("reasoning_none", True),
            reasoning_none_alias=d.get("reasoning_none_alias"),
            thinking_template_kwarg=d.get("thinking_template_kwarg", False),
            vision=d.get("vision", False),
            tools=d.get("tools", True),
            strip_think=d.get("strip_think", False),
            system_first_only=d.get("system_first_only", False),
            max_output=d.get("max_output"),
            max_input_chars=d.get("max_input_chars", 15_000),
        )
        deployments[name] = Deployment(
            name,
            d["provider"],
            d["model"],
            provider["base_url"],
            os.getenv(provider["key_env"]),
            rpm=_scaled(models_config._env_int(f"LLM_{env_name}_RPM", d.get("rpm"))),
            # TPM is left unscaled: Groq also applies it as a hard per-request
            # ceiling (prompt + max_tokens ≤ TPM, else 413), and a 15K-char
            # chunk with a 4K answer only just fits 8K. Scaling it would make
            # that request "never fits" and skip Groq forever.
            tpm=models_config._env_int(f"LLM_{env_name}_TPM", d.get("tpm")),
            rpd=_scaled(models_config._env_int(f"LLM_{env_name}_RPD", d.get("rpd"))),
            pools=pools[d["provider"]],
            caps=caps,
            disabled=name in models_config.LLM_DISABLE,
        )
    return deployments


_DEPLOYMENTS: dict[str, Deployment] = _build()
_CLIENTS: dict[str, AsyncOpenAI] = {}
_RECENT: deque[dict] = deque(maxlen=50)


def deployment(name: str) -> Deployment:
    return _DEPLOYMENTS[name]


def deployments_for(tier: str) -> list[Deployment]:
    tier = models_config.TIER_ALIASES.get(tier, tier)
    names = models_config.TIERS.get(tier)
    if names is None:
        raise KeyError(f"unknown LLM tier {tier!r}; known: {sorted(models_config.TIERS)}")
    return [_DEPLOYMENTS[n] for n in names if n in _DEPLOYMENTS]


def is_configured() -> bool:
    """At least one deployment has an API key. Replaces the old `if not groq_client`."""
    return any(d.enabled() for d in _DEPLOYMENTS.values())


def chunk_chars(tier: str) -> int:
    """How much text to hand the tier per request: the first usable deployment's cap.

    Gemini takes 60K characters a chunk where Groq took 15K, which is four
    times fewer calls for the same PDF — as long as Gemini is actually
    available right now, hence the budget check.
    """
    now = time.monotonic()
    deps = deployments_for(tier)
    for d in deps:
        ok, _, _ = d.check(now, 1)
        if ok:
            return d.caps.max_input_chars
    return min((d.caps.max_input_chars for d in deps), default=15_000)


def _client(dep: Deployment) -> AsyncOpenAI:
    client = _CLIENTS.get(dep.provider)
    if client is None:
        client = AsyncOpenAI(
            base_url=dep.base_url,
            api_key=dep.api_key,
            max_retries=0,
            timeout=models_config.LLM_TIMEOUT_S,
        )
        _CLIENTS[dep.provider] = client
    return client


# --- Text helpers -----------------------------------------------------------

_THINK_CLOSED = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN = re.compile(r"<think>.*$", re.DOTALL)


def strip_think(text: str | None) -> str:
    """Remove qwen-style reasoning blocks, including one cut off by max_tokens."""
    if not text:
        return ""
    text = _THINK_CLOSED.sub("", text)
    text = _THINK_OPEN.sub("", text)
    return text.strip()


_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$")


def parse_json(text: str | None) -> Any:
    """json.loads that survives <think> blocks, fences and chatter around the object.

    Needed because not every deployment has a real JSON mode (NVIDIA), and
    even those that do sometimes wrap the object in a fence.
    """
    s = strip_think(text)
    s = _FENCE.sub("", s).strip()
    if not s:
        raise json.JSONDecodeError("empty response", s, 0)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    starts = [(s.find(o), o, c) for o, c in (("{", "}"), ("[", "]")) if s.find(o) != -1]
    for _, open_, close in sorted(starts):
        end = s.rfind(close)
        start = s.find(open_)
        if end > start:
            try:
                return json.loads(s[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise json.JSONDecodeError("no JSON value found", s, 0)


def _has_image(messages: list) -> bool:
    for m in messages or []:
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "image", "input_image"):
                    return True
    return False


# --- Request normalisation --------------------------------------------------

_JSON_NUDGE = "\n\nReturn ONLY a valid JSON value. No prose, no markdown fences."


def _normalise(dep: Deployment, payload: dict) -> dict:
    """Shape one call's kwargs to what this deployment accepts.

    Call sites pass what they always passed; the differences between
    providers live here and nowhere else.
    """
    caps = dep.caps
    payload.pop("model", None)

    wanted_no_reasoning = payload.get("reasoning_effort") == "none"
    if wanted_no_reasoning and not caps.reasoning_none:
        if caps.reasoning_none_alias:
            payload["reasoning_effort"] = caps.reasoning_none_alias
        else:
            payload.pop("reasoning_effort")
        if caps.thinking_template_kwarg:
            payload.setdefault("extra_body", {})["chat_template_kwargs"] = {"enable_thinking": False}

    messages = [dict(m) if isinstance(m, dict) else m for m in payload.get("messages") or []]

    has_tools = bool(payload.get("tools"))
    if has_tools and not caps.tools:
        raise ValueError(f"{dep.name} does not support tools")

    if "response_format" in payload:
        json_ok = caps.json_mode and not (has_tools and dep.provider == "groq")
        if not json_ok:
            payload.pop("response_format")
            for i in range(len(messages) - 1, -1, -1):
                m = messages[i]
                if isinstance(m, dict) and m.get("role") == "user":
                    content = m.get("content")
                    if isinstance(content, str):
                        m["content"] = content + _JSON_NUDGE
                    elif isinstance(content, list):
                        m["content"] = list(content) + [{"type": "text", "text": _JSON_NUDGE.strip()}]
                    break

    if caps.max_output and payload.get("max_tokens"):
        payload["max_tokens"] = min(int(payload["max_tokens"]), caps.max_output)

    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls") and m.get("content") is None:
            m["content"] = ""
        elif role == "tool":
            # `name` on a tool message is a Groq-ism; the strict endpoints reject it.
            m.pop("name", None)
        elif role == "system" and i > 0 and caps.system_first_only:
            m["role"] = "user"
    payload["messages"] = messages
    return payload


# --- Failure classification -------------------------------------------------

_INCOMPATIBLE = re.compile(
    r"not supported|unsupported|unknown (?:name|field|parameter)|does not support|"
    r"model_not_found|not found|invalid (?:parameter|argument|value)",
    re.I,
)


def _seconds_until_utc_midnight() -> float:
    now = datetime.now(timezone.utc)
    tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() + 86_400
    return max(60.0, tomorrow - now.timestamp())


def _classify(err: Exception) -> tuple[str, float, str]:
    """(kind, cooldown_seconds, label). kind ∈ transient | incompatible | fatal."""
    status = getattr(err, "status_code", None)
    text = str(err)
    if isinstance(err, RateLimitError) or status == 429:
        if mentions_daily_quota(err):
            return "transient", _seconds_until_utc_midnight(), "daily_quota"
        return "transient", parse_retry_delay(err) or 15.0, "rate_limited"
    if status == 402:
        return "transient", 6 * 3600.0, "no_credits"
    if status == 413:
        # This request is too big for this deployment; the next one may take
        # it. Nothing is wrong with the deployment, so no cooldown.
        return "transient", 0.0, "too_large"
    if status in (500, 502, 503, 504, 529):
        return "transient", 30.0, f"http_{status}"
    if isinstance(err, (APIConnectionError, APITimeoutError, asyncio.TimeoutError)):
        # Longer than LLM_MAX_WAIT_S on purpose: a timed-out deployment must
        # not be waited for and retried inside the same call (that turned one
        # slow provider into a 4.5-minute request when it was the only one left).
        return "transient", 30.0, "connection"
    if status in (404, 410) or (status == 400 and _INCOMPATIBLE.search(text)):
        # Model retired (NVIDIA returns 410 "end of life") or a parameter it
        # does not accept. Retrying in a minute will not help; an hour later
        # someone may have fixed the config.
        return "incompatible", 3600.0, "incompatible"
    if status in (400, 401, 403, 422):
        return "fatal", 0.0, f"http_{status}"
    if isinstance(err, APIStatusError):
        return "transient", 15.0, f"http_{status}"
    # Something that is not an HTTP failure (LangChain parsing, a bug). Let the
    # next deployment have a go but keep the error visible.
    return "transient", 30.0, "unknown"


def _is_empty(result: Any) -> bool:
    choices = getattr(result, "choices", None)
    if choices is not None:
        if not choices:
            return True
        message = choices[0].message
        content = message.content
        return not (isinstance(content, str) and content.strip()) and not getattr(message, "tool_calls", None)
    content = getattr(result, "content", None)
    tool_calls = getattr(result, "tool_calls", None)
    if content is None and tool_calls is None:
        return False
    has_text = bool(content.strip()) if isinstance(content, str) else bool(content)
    return not has_text and not tool_calls


def _record(**fields: Any) -> None:
    fields["at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _RECENT.append(fields)


# --- The loop ---------------------------------------------------------------

async def run(
    tier: str,
    est_tokens: int | Callable[[Deployment], int],
    fn: Callable[[Deployment], Awaitable[T]],
    usage_of: Callable[[T], int | None] | None = None,
    *,
    needs_vision: bool = False,
    needs_tools: bool = False,
) -> T:
    """Call `fn(deployment)` on the first deployment in `tier` with budget to spare.

    Sweeps the tier in order; a deployment whose window is full or that is
    cooling down is skipped, one that fails is cooled and skipped. If a whole
    sweep fails and the soonest deployment frees up within LLM_MAX_WAIT_S,
    waits for it and sweeps again (three sweeps at most). Otherwise raises
    LLMUnavailable. A fatal error (bad request, bad key) propagates at once
    because retrying a bad request never succeeds.

    `est_tokens` may be a callable so the estimate can depend on the
    deployment (its max_output clamp changes how much output is reserved).
    """
    estimate = est_tokens if callable(est_tokens) else (lambda _dep: est_tokens)
    deps = deployments_for(tier)
    if needs_vision:
        deps = [d for d in deps if d.caps.vision]
    if needs_tools:
        deps = [d for d in deps if d.caps.tools]
    if not deps:
        raise LLMUnavailable(f"{tier}: no deployment configured for this request")

    reasons: dict[str, str] = {}
    last_error: Exception | None = None
    for sweep in range(3):
        min_wait = math.inf
        for dep in deps:
            now = time.monotonic()
            est = int(estimate(dep))
            ok, wait, reason = dep.check(now, est)
            if not ok:
                dep.skipped += 1
                reasons[dep.name] = reason
                min_wait = min(min_wait, wait)
                logger.info("llm tier=%s dep=%s skip reason=%s wait=%s est=%d", tier, dep.name, reason,
                            "inf" if math.isinf(wait) else f"{wait:.1f}s", est)
                continue

            reservation = dep.reserve(now, est)
            started = time.monotonic()
            try:
                result = await fn(dep)
            except Exception as err:
                ms = int((time.monotonic() - started) * 1000)
                dep.settle(reservation, 0)
                dep.failed += 1
                dep.last_error = str(err)[:200]
                last_error = err
                kind, cooldown, label = _classify(err)
                reasons[dep.name] = label
                _record(tier=tier, dep=dep.name, status="error", label=label, ms=ms)
                if kind == "fatal":
                    logger.error("llm tier=%s dep=%s status=fatal label=%s err=%s", tier, dep.name, label, str(err)[:300])
                    raise
                dep.cool(time.monotonic(), cooldown, label)
                min_wait = min(min_wait, cooldown)
                logger.warning("llm tier=%s dep=%s status=error label=%s cool=%.0fs sweep=%d err=%s",
                               tier, dep.name, label, cooldown, sweep + 1, str(err)[:300],
                               exc_info=label == "unknown")
                continue

            ms = int((time.monotonic() - started) * 1000)
            used = None
            if usage_of is not None:
                try:
                    used = usage_of(result)
                except Exception:
                    used = None
            dep.settle(reservation, used if used else est)

            if _is_empty(result):
                dep.cool(time.monotonic(), 5.0, "empty")
                dep.failed += 1
                reasons[dep.name] = "empty"
                min_wait = min(min_wait, 5.0)
                _record(tier=tier, dep=dep.name, status="empty", ms=ms)
                logger.warning("llm tier=%s dep=%s status=empty sweep=%d", tier, dep.name, sweep + 1)
                continue

            dep.served += 1
            _record(tier=tier, dep=dep.name, status="ok", tokens=used or est, ms=ms)
            logger.info("llm tier=%s dep=%s model=%s status=ok tokens=%s est=%s ms=%d sweep=%d",
                        tier, dep.name, dep.model, used if used else "?", est, ms, sweep + 1)
            return result

        if sweep < 2 and min_wait <= models_config.LLM_MAX_WAIT_S:
            logger.info("llm tier=%s all deployments busy, waiting %.1fs (%s)", tier, min_wait, reasons)
            await asyncio.sleep(min_wait)
        else:
            break

    detail = ", ".join(f"{k}={v}" for k, v in reasons.items()) or "none configured"
    if last_error is not None:
        detail += f"; last error: {str(last_error)[:200]}"
    raise LLMUnavailable(f"{tier}: all deployments unavailable ({detail})")


async def chat(tier: str, **kwargs: Any):
    """Drop-in for `client.chat.completions.create(model=..., **kwargs)` minus `model`."""
    messages = kwargs.get("messages") or []
    max_tokens = kwargs.get("max_tokens")

    def est(dep: Deployment) -> int:
        # Reserve the output budget this deployment will actually be asked
        # for, i.e. after its max_output clamp, not the caller's ceiling.
        reserved = max_tokens
        if reserved and dep.caps.max_output:
            reserved = min(int(reserved), dep.caps.max_output)
        return estimate_tokens(messages, reserved, kwargs.get("tools"))

    async def call(dep: Deployment):
        payload = _normalise(dep, dict(kwargs))
        response = await _client(dep).chat.completions.create(model=dep.model, **payload)
        if response.choices:
            message = response.choices[0].message
            if isinstance(message.content, str) and (dep.caps.strip_think or "<think>" in message.content):
                message.content = strip_think(message.content)
        return response

    def usage_of(response) -> int | None:
        usage = getattr(response, "usage", None)
        return getattr(usage, "total_tokens", None) if usage else None

    return await run(
        tier, est, call, usage_of,
        needs_vision=_has_image(messages),
        needs_tools=bool(kwargs.get("tools")),
    )


def snapshot() -> dict:
    """What /api/health/llm shows: budgets, cooldowns and the last 50 calls."""
    now = time.monotonic()
    return {
        "tiers": {
            models_config.TIER_ALIASES.get(t, t): [d.name for d in deployments_for(t)]
            for t in models_config.TIERS
        },
        "safety": models_config.LLM_SAFETY,
        "deployments": {name: dep.state(now) for name, dep in _DEPLOYMENTS.items()},
        "recent": list(_RECENT),
    }
