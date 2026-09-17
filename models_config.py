"""Groq model ids, in one place.

Groq retires model ids without warning — on 2026-09-08 all three ids this
codebase used (`qwen/qwen3-32b`, `llama-3.1-8b-instant`,
`meta-llama/llama-4-scout-17b-16e-instruct`) disappeared at once and every
LLM-backed feature started returning `model_not_found`. Keeping the ids here
means the next retirement is a one-line fix, and each can be overridden from
.env without touching code.

Check what is currently available with:
    curl -H "Authorization: Bearer $GROQ_API_KEY" \
         https://api.groq.com/openai/v1/models
"""
import os

# Reasoning and generation: answers, grading, Q&A extraction, STAR drafts.
TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "qwen/qwen3.8-27b")

# Short classification and keyword work, where latency matters more than depth.
FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")

# Image understanding, for pulling Q&A pairs out of screenshots.
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")

# Second choice when the first is over capacity. The free tier returns 503
# "over capacity" often enough that one model is not enough to rely on.
VISION_FALLBACK_MODEL = os.getenv("GROQ_VISION_FALLBACK_MODEL", "qwen/qwen3.6-27b")

# Models that can read images, in preference order.
VISION_MODELS = [m for m in (VISION_MODEL, VISION_FALLBACK_MODEL) if m]
