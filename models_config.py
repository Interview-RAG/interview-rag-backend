"""Every LLM provider, model id and rate limit, in one place.

Providers retire model ids without warning — on 2026-09-08 Groq dropped all
three ids this codebase used at once, and on 2026-09-23 `qwen/qwen3.6-27b`
was gone too. Keeping the ids here means the next retirement is a one-line
fix, and everything can be overridden from .env without touching code.

Check what is currently available with:
    curl -H "Authorization: Bearer $GROQ_API_KEY"   https://api.groq.com/openai/v1/models
    curl -H "Authorization: Bearer $GEMINI_API_KEY" https://generativelanguage.googleapis.com/v1beta/openai/models
    curl -H "Authorization: Bearer $NVIDIA_API_KEY" https://integrate.api.nvidia.com/v1/models

Limits below are the published free-tier numbers as of 2026-09-23. The router
in `llm.py` spends only `LLM_SAFETY` (85%) of the request and daily limits so
a little drift on the provider's side does not turn into 429s. Tokens-per-
minute is used as published, because Groq also enforces it per request
(prompt + max_tokens must fit) and a scaled figure would refuse work that
fits. A deployment whose provider key is missing is skipped, so the app still
runs with a single key configured.
"""
import os


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return [x.strip() for x in raw.split(",") if x.strip()]


# --- Providers: one OpenAI-compatible endpoint each --------------------------
# `tpd` is an org-wide daily token pool shared by every deployment on that
# provider (Groq's is the one that actually bites).
PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "tpd": _env_int("LLM_GROQ_TPD", 200_000),
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "key_env": "NVIDIA_API_KEY",
    },
}

# --- Deployments: (provider, model) with limits and capabilities -------------
# Per-deployment limits can be overridden with LLM_<NAME>_RPM / _TPM / _RPD,
# where NAME is the key upper-cased with '-' → '_' (e.g. LLM_GEMINI_FLASH_RPM).
DEPLOYMENTS = {
    # Gemini 2.5 Flash: 10 RPM, 250K TPM, ~1,500 RPD, no daily token cap.
    # The workhorse: huge context, JSON mode, tools and vision all work.
    "gemini-flash": dict(
        provider="gemini",
        model=os.getenv("GEMINI_HEAVY_MODEL", "gemini-2.5-flash"),
        rpm=10, tpm=250_000, rpd=1_500,
        json_mode=True, reasoning_none=True, vision=True, tools=True,
        system_first_only=True, max_output=16_384, max_input_chars=60_000,
    ),
    # Gemini 2.5 Flash-Lite: 15 RPM, 250K TPM, 1,000 RPD. Cheap classifier.
    "gemini-lite": dict(
        provider="gemini",
        model=os.getenv("GEMINI_LITE_MODEL", "gemini-2.5-flash-lite"),
        rpm=15, tpm=250_000, rpd=1_000,
        json_mode=True, reasoning_none=True, vision=True, tools=True,
        system_first_only=True, max_output=8_192, max_input_chars=60_000,
    ),
    # Groq qwen3.8-27b: 30 RPM, 8K TPM, 1K RPD, shares the 200K/day pool.
    # Only Groq model left that reads images. Emits <think> blocks.
    "groq-text": dict(
        provider="groq",
        model=os.getenv("GROQ_TEXT_MODEL", "qwen/qwen3.8-27b"),
        rpm=30, tpm=8_000, rpd=1_000,
        json_mode=True, reasoning_none=True, vision=True, tools=True,
        strip_think=True, max_output=4_096, max_input_chars=15_000,
    ),
    # Groq gpt-oss-20b: same limits. Fast, but only accepts
    # reasoning_effort low|medium|high, so "none" is sent as "low".
    "groq-fast": dict(
        provider="groq",
        model=os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b"),
        rpm=30, tpm=8_000, rpd=1_000,
        json_mode=True, reasoning_none=False, reasoning_none_alias="low",
        vision=False, tools=True, max_output=4_096, max_input_chars=15_000,
    ),
    # NVIDIA NIM Nemotron 3 Super: 40 RPM, no published daily cap. Overflow
    # capacity. NIM retired openai/gpt-oss-120b on 2026-09-03 (HTTP 410), so
    # the default is NVIDIA's own model, the least likely to vanish there.
    # JSON mode varies by model on NIM, so we ask for JSON in the prompt; the
    # `reasoning_effort` parameter is not accepted, thinking is switched off
    # through the chat template instead.
    "nvidia-text": dict(
        provider="nvidia",
        model=os.getenv("NVIDIA_TEXT_MODEL", "nvidia/nemotron-3-super-120b-a12b"),
        rpm=40, tpm=None, rpd=None,
        json_mode=False, reasoning_none=False, reasoning_none_alias=None,
        thinking_template_kwarg=True,
        vision=False, tools=True, strip_think=True,
        max_output=8_192, max_input_chars=40_000,
    ),
}

# Optional second Groq vision model. Off by default because the previous
# default (qwen/qwen3.6-27b) has been retired.
if os.getenv("GROQ_VISION_FALLBACK_MODEL"):
    DEPLOYMENTS["groq-vision-fb"] = dict(
        provider="groq",
        model=os.environ["GROQ_VISION_FALLBACK_MODEL"],
        rpm=30, tpm=8_000, rpd=1_000,
        json_mode=True, reasoning_none=True, vision=True, tools=False,
        strip_think=True, max_output=4_096, max_input_chars=15_000,
    )

# --- Tiers: which deployments serve which kind of work, in preference order --
# Override any tier with LLM_TIER_<NAME>="dep-a,dep-b".
#   heavy  – long prompts or careful output: PDF/resume parsing, grading, STAR
#   fast   – tiny prompts where latency matters: yes/no, tags, keywords
#   vision – anything carrying an image
#   agent  – the LangGraph chat agent (tool calling, long history)
_VISION_DEFAULT = ["gemini-flash", "groq-text"] + (
    ["groq-vision-fb"] if "groq-vision-fb" in DEPLOYMENTS else []
)
TIERS = {
    "heavy": _env_list("LLM_TIER_HEAVY", ["gemini-flash", "nvidia-text", "groq-text"]),
    "fast": _env_list("LLM_TIER_FAST", ["groq-fast", "gemini-lite", "nvidia-text"]),
    "vision": _env_list("LLM_TIER_VISION", _VISION_DEFAULT),
    "agent": _env_list("LLM_TIER_AGENT", ["gemini-flash", "nvidia-text", "groq-text"]),
}
TIER_ALIASES = {"text": "heavy"}

# --- Router knobs -------------------------------------------------------------
# Fraction of each published limit we allow ourselves to spend.
LLM_SAFETY = float(os.getenv("LLM_SAFETY", "0.85"))
# If every deployment is busy, wait for the soonest one only if it frees up
# within this many seconds; otherwise fail fast with LLMUnavailable.
LLM_MAX_WAIT_S = float(os.getenv("LLM_MAX_WAIT_S", "20"))
# Per-request HTTP timeout. Big PDF chunks on Gemini can take a while.
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "90"))
# Comma-separated deployment names to switch off without editing the tables.
LLM_DISABLE = set(_env_list("LLM_DISABLE", []))

# Agent output ceiling. Interview answers run a few hundred tokens; this is
# also the amount reserved from the token budget before each agent turn.
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "1500"))
