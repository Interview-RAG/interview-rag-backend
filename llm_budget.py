"""Budget accounting for the LLM router: rolling windows, per-deployment state.

Pure Python, no network, no module-level state, so it can be unit-tested with
a fake clock. `llm.py` owns the live instances.

Why this exists: every free tier publishes its limits (requests and tokens per
minute, per day). Spending the budget *before* a request goes out means we
mostly never see a 429, instead of discovering the limit by tripping it and
then burning the rest of the minute on retries.
"""
from __future__ import annotations

import json
import math
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any


class Window:
    """Rolling-window counter.

    `limit=None` means unlimited. Entries are `[timestamp, count]` lists so a
    caller can reserve an *estimate* and correct it to the real usage later
    without touching the window's internals.
    """

    def __init__(self, seconds: float, limit: int | None):
        self.seconds = float(seconds)
        self.limit = limit
        self._entries: deque[list] = deque()

    def _prune(self, now: float) -> None:
        cutoff = now - self.seconds
        while self._entries and self._entries[0][0] <= cutoff:
            self._entries.popleft()

    def used(self, now: float) -> int:
        self._prune(now)
        return sum(e[1] for e in self._entries)

    def wait_for(self, now: float, n: int) -> float:
        """Seconds until `n` more units fit. 0.0 if they fit now, `inf` if never."""
        if self.limit is None:
            return 0.0
        if n > self.limit:
            return math.inf
        self._prune(now)
        used = sum(e[1] for e in self._entries)
        if used + n <= self.limit:
            return 0.0
        freed = 0
        for ts, count in self._entries:
            freed += count
            if used - freed + n <= self.limit:
                # Small pad so the caller lands just after the entry expires.
                return max(0.0, ts + self.seconds - now) + 0.05
        return math.inf

    def add(self, now: float, n: int) -> list:
        entry = [now, n]
        self._entries.append(entry)
        return entry


@dataclass
class Caps:
    """What a deployment can accept. Drives request normalisation in llm.py."""

    json_mode: bool = True            # accepts response_format={"type": "json_object"}
    reasoning_none: bool = True       # accepts reasoning_effort="none"
    reasoning_none_alias: str | None = None  # what to send instead of "none" (e.g. "low" for gpt-oss)
    thinking_template_kwarg: bool = False    # turn thinking off via chat_template_kwargs (NIM qwen/nemotron)
    vision: bool = False              # accepts image_url parts
    tools: bool = True                # accepts function tools
    strip_think: bool = False         # emits <think> blocks that must be removed
    system_first_only: bool = False   # only the first message may be role=system
    max_output: int | None = None     # clamp max_tokens to this
    max_input_chars: int = 15_000     # how big a chunk of text to hand it


@dataclass
class Reservation:
    request_entries: list = field(default_factory=list)
    token_entries: list = field(default_factory=list)


class Deployment:
    """One (provider, model) pair with its own budget and health."""

    def __init__(
        self,
        name: str,
        provider: str,
        model: str,
        base_url: str,
        api_key: str | None,
        *,
        rpm: int | None,
        tpm: int | None,
        rpd: int | None,
        pools: list[tuple[str, Window]] | None = None,
        caps: Caps | None = None,
        disabled: bool = False,
    ):
        self.name = name
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.api_key = api_key or None
        self.rpm = Window(60, rpm)
        self.tpm = Window(60, tpm)
        self.rpd = Window(86_400, rpd)
        self.pools = pools or []          # shared org-level pools, e.g. Groq tokens/day
        self.caps = caps or Caps()
        self.disabled = disabled
        self.cooldown_until = 0.0
        self.cooldown_reason = ""
        self.served = 0
        self.failed = 0
        self.skipped = 0
        self.last_error = ""

    def enabled(self) -> bool:
        return bool(self.api_key) and not self.disabled

    def check(self, now: float, est_tokens: int) -> tuple[bool, float, str]:
        """(ok, seconds_to_wait, reason). Reasons are short slugs for logs."""
        if self.disabled:
            return False, math.inf, "disabled"
        if not self.api_key:
            return False, math.inf, "no_key"
        if self.cooldown_until > now:
            return False, self.cooldown_until - now, f"cooldown:{self.cooldown_reason}"

        checks = [("rpd", self.rpd, 1)]
        checks += [(name, win, est_tokens) for name, win in self.pools]
        checks += [("rpm", self.rpm, 1), ("tpm", self.tpm, est_tokens)]
        for name, window, n in checks:
            wait = window.wait_for(now, n)
            if wait > 0:
                return False, wait, name
        return True, 0.0, ""

    def reserve(self, now: float, est_tokens: int) -> Reservation:
        res = Reservation()
        res.request_entries.append(self.rpm.add(now, 1))
        res.request_entries.append(self.rpd.add(now, 1))
        res.token_entries.append(self.tpm.add(now, est_tokens))
        for _, pool in self.pools:
            res.token_entries.append(pool.add(now, est_tokens))
        return res

    def settle(self, res: Reservation, actual_tokens: int | None) -> None:
        """Replace the token estimate with what the provider reported."""
        if actual_tokens is None:
            return
        for entry in res.token_entries:
            entry[1] = int(actual_tokens)

    def cool(self, now: float, seconds: float, reason: str) -> None:
        if seconds <= 0:
            return
        until = now + seconds
        if until > self.cooldown_until:
            self.cooldown_until = until
            self.cooldown_reason = reason

    def state(self, now: float) -> dict[str, Any]:
        def pair(window: Window) -> dict:
            return {"used": window.used(now), "limit": window.limit}

        return {
            "provider": self.provider,
            "model": self.model,
            "enabled": self.enabled(),
            "cooldown_s": round(max(0.0, self.cooldown_until - now), 1),
            "cooldown_reason": self.cooldown_reason if self.cooldown_until > now else "",
            "rpm": pair(self.rpm),
            "tpm": pair(self.tpm),
            "rpd": pair(self.rpd),
            "pools": {name: pair(win) for name, win in self.pools},
            "served": self.served,
            "failed": self.failed,
            "skipped": self.skipped,
            "last_error": self.last_error,
        }


# --- Estimation and error parsing -------------------------------------------

def _content_size(content: Any) -> tuple[int, int]:
    """(characters, images) in one message's content."""
    if content is None:
        return 0, 0
    if isinstance(content, str):
        return len(content), 0
    if isinstance(content, list):
        chars, images = 0, 0
        for part in content:
            if isinstance(part, dict):
                kind = part.get("type")
                if kind == "text":
                    chars += len(part.get("text") or "")
                elif kind in ("image_url", "image", "input_image"):
                    images += 1
                else:
                    chars += len(json.dumps(part, default=str))
            else:
                chars += len(str(part))
        return chars, images
    return len(str(content)), 0


def estimate_tokens(messages: list, max_tokens: int | None, tools: list | None = None) -> int:
    """Rough token count for budgeting: ~4 chars per token, plus reserved output.

    Accepts OpenAI-style dicts or LangChain message objects. Overestimating a
    little is fine; `settle()` fixes the number once the provider reports it.
    """
    chars = 0
    images = 0
    for m in messages or []:
        if isinstance(m, dict):
            content = m.get("content")
            extra = m.get("tool_calls")
        else:
            content = getattr(m, "content", None)
            extra = getattr(m, "tool_calls", None)
        c, i = _content_size(content)
        chars += c
        images += i
        if extra:
            chars += len(json.dumps(extra, default=str))
    tokens = chars / 4 + 4 * len(messages or [])
    tokens += 1000 * images
    if tools:
        tokens += len(json.dumps(tools, default=str)) / 4
    tokens += max_tokens if max_tokens else 1024
    return int(math.ceil(tokens))


_DELAY_PATTERNS = (
    # Groq: "Please try again in 2.5s" / "in 1m3.2s" / "in 250ms"
    re.compile(r"try again in (?:(\d+)m)?([0-9.]+)\s*(ms|s)\b", re.I),
    # Gemini: "Please retry in 23.4s" and {"retryDelay": "23s"}
    re.compile(r"retry in ([0-9.]+)\s*s\b", re.I),
    re.compile(r'"retryDelay"\s*:\s*"([0-9.]+)s"', re.I),
)


def _error_text(err: Exception) -> str:
    parts = [str(err)]
    body = getattr(err, "body", None)
    if body:
        try:
            parts.append(json.dumps(body, default=str))
        except Exception:
            parts.append(str(body))
    return " ".join(parts)


def parse_retry_delay(err: Exception) -> float | None:
    """How long the provider asked us to wait, in seconds, if it said."""
    response = getattr(err, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        ms = headers.get("retry-after-ms")
        if ms:
            try:
                return float(ms) / 1000.0
            except ValueError:
                pass
        ra = headers.get("retry-after")
        if ra:
            try:
                return float(ra)
            except ValueError:
                pass  # HTTP-date form; fall through to the body

    text = _error_text(err)
    m = _DELAY_PATTERNS[0].search(text)
    if m:
        minutes = float(m.group(1) or 0)
        value = float(m.group(2))
        if m.group(3).lower() == "ms":
            value /= 1000.0
        return minutes * 60 + value + 0.5
    for pattern in _DELAY_PATTERNS[1:]:
        m = pattern.search(text)
        if m:
            return float(m.group(1)) + 0.5
    return None


_DAILY_MARKERS = re.compile(
    r"per day|tokens per day|requests per day|\bTPD\b|\bRPD\b|PerDay|daily", re.I
)


def mentions_daily_quota(err: Exception) -> bool:
    return bool(_DAILY_MARKERS.search(_error_text(err)))
