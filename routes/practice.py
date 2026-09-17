import json
import asyncio
import re
from datetime import date, timedelta

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from auth import get_current_user
import database
import rag
import models_config

router = APIRouter(prefix="/api/practice", tags=["practice"])

# The intervals the Practice screen advertises on its rating buttons.
# Again resets the ladder; Good and Easy step it up multiplicatively.
EASE_STEPS = {
    "again": 1,
    "good": 4,
    "easy": 10,
}
EASE_MULTIPLIER = {
    "again": 0.0,
    "good": 2.0,
    "easy": 2.6,
}


def next_interval(ease: str, current_interval: int) -> int:
    """Days until this card should come back."""
    base = EASE_STEPS[ease]
    if ease == "again" or not current_interval:
        return base
    grown = int(round(current_interval * EASE_MULTIPLIER[ease]))
    return max(base, min(grown, 365))


class ReviewIn(BaseModel):
    ease: str
    mode: str = "flip"
    score: int | None = None


@router.get("/due")
async def get_due(user_id: str = Depends(get_current_user)):
    """Cards at or past their due date, oldest due first."""
    today = date.today().isoformat()

    def fetch():
        return (
            database.supabase.table("qa_records")
            .select("*")
            .eq("user_id", user_id)
            .lte("due_date", today)
            .order("due_date")
            .execute()
        )

    resp = await asyncio.to_thread(fetch)
    return [_shape(r) for r in (resp.data or [])]


@router.post("/{qa_id}/review")
async def review_card(
    qa_id: int, body: ReviewIn, user_id: str = Depends(get_current_user)
):
    """Record a rating, advance the schedule, and log it for Progress."""
    if body.ease not in EASE_STEPS:
        raise HTTPException(status_code=400, detail="ease must be again, good or easy")
    if body.mode not in ("flip", "type"):
        raise HTTPException(status_code=400, detail="mode must be flip or type")

    def fetch():
        return (
            database.supabase.table("qa_records")
            .select("*")
            .eq("id", qa_id)
            .eq("user_id", user_id)
            .execute()
        )

    existing = await asyncio.to_thread(fetch)
    if not existing.data:
        raise HTTPException(status_code=404, detail="Q&A not found")

    record = existing.data[0]
    interval = next_interval(body.ease, record.get("interval_days") or 0)
    due = (date.today() + timedelta(days=interval)).isoformat()

    def update():
        return (
            database.supabase.table("qa_records")
            .update(
                {
                    "due_date": due,
                    "interval_days": interval,
                    "review_count": (record.get("review_count") or 0) + 1,
                    "last_ease": body.ease,
                }
            )
            .eq("id", qa_id)
            .eq("user_id", user_id)
            .execute()
        )

    updated = await asyncio.to_thread(update)

    def log():
        return (
            database.supabase.table("qa_reviews")
            .insert(
                {
                    "qa_id": qa_id,
                    "user_id": user_id,
                    "ease": body.ease,
                    "score": body.score,
                    "mode": body.mode,
                }
            )
            .execute()
        )

    try:
        await asyncio.to_thread(log)
    except Exception as e:
        # A missing history row must not lose the schedule update.
        print(f"qa_reviews insert failed: {e}")

    return {"interval_days": interval, "due_date": due, "qa": _shape(updated.data[0])}


class GradeIn(BaseModel):
    typed_answer: str


@router.post("/{qa_id}/grade")
async def grade_answer(
    qa_id: int, body: GradeIn, user_id: str = Depends(get_current_user)
):
    """Score a typed answer against the saved one: 0-10 plus what was missed."""
    if not body.typed_answer.strip():
        raise HTTPException(status_code=400, detail="Answer is empty")
    if not rag.groq_client:
        raise HTTPException(status_code=503, detail="Grading is unavailable")

    def fetch():
        return (
            database.supabase.table("qa_records")
            .select("*")
            .eq("id", qa_id)
            .eq("user_id", user_id)
            .execute()
        )

    existing = await asyncio.to_thread(fetch)
    if not existing.data:
        raise HTTPException(status_code=404, detail="Q&A not found")

    record = existing.data[0]
    questions = record["questions_json"]
    if isinstance(questions, str):
        questions = json.loads(questions)
    question = questions[0] if questions else ""

    prompt = (
        "You are an interview coach grading a candidate's spoken answer against "
        "the reference answer they saved earlier.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{record['answer']}\n\n"
        f"CANDIDATE ANSWER:\n{body.typed_answer}\n\n"
        "Reply with JSON only, no prose and no markdown fences:\n"
        '{"score": <integer 0-10>, "verdict": "<one short sentence naming what '
        'they got right>", "missed": "<one or two sentences on what was missing '
        'and what would raise the score>"}'
    )

    try:
        completion = await rag.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=models_config.TEXT_MODEL,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content
        # qwen can wrap its reply in <think> blocks or fences.
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE)
        data = json.loads(raw.strip())
    except Exception as e:
        print(f"Grading failed: {e}")
        raise HTTPException(status_code=502, detail="Could not grade that answer")

    try:
        score = max(0, min(10, int(data.get("score", 0))))
    except (TypeError, ValueError):
        score = 0

    return {
        "score": score,
        "verdict": str(data.get("verdict") or "").strip(),
        "missed": str(data.get("missed") or "").strip(),
    }


def _shape(r: dict) -> dict:
    """One row of qa_records as the frontend expects it."""
    questions = r["questions_json"]
    if isinstance(questions, str):
        questions = json.loads(questions)
    return {
        "id": r["id"],
        "questions": questions,
        "answer": r["answer"],
        "tags": r.get("tags") or [],
        "due_date": r.get("due_date"),
        "review_count": r.get("review_count") or 0,
        "last_ease": r.get("last_ease"),
        "interval_days": r.get("interval_days") or 0,
    }
