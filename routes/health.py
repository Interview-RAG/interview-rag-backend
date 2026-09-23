"""Operational visibility into the LLM router.

GET /api/health/llm shows, per deployment, how much of each free-tier window
is spent, whether it is cooling down and why, and the last 50 calls with the
deployment that served each. Authenticated like everything else; the payload
carries no keys.
"""
from fastapi import APIRouter, Depends

from auth import get_current_user
import llm

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("/llm")
async def llm_health(user_id: str = Depends(get_current_user)):
    return llm.snapshot()
