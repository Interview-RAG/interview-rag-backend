import json
import asyncio
from collections import Counter
from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends

from auth import get_current_user
import database

router = APIRouter(prefix="/api/progress", tags=["progress"])

BAR_DAYS = 14


@router.get("")
async def get_progress(user_id: str = Depends(get_current_user)):
    """Everything the Progress screen renders, in one round trip."""
    today = date.today()

    def fetch_records():
        return (
            database.supabase.table("qa_records")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )

    def fetch_reviews():
        since = (today - timedelta(days=120)).isoformat()
        return (
            database.supabase.table("qa_reviews")
            .select("qa_id, ease, score, reviewed_at")
            .eq("user_id", user_id)
            .gte("reviewed_at", since)
            .order("reviewed_at", desc=True)
            .execute()
        )

    def fetch_resume():
        return (
            database.supabase.table("user_resumes")
            .select("parsed_data")
            .eq("user_id", user_id)
            .execute()
        )

    records_resp, reviews_resp, resume_resp = await asyncio.gather(
        asyncio.to_thread(fetch_records),
        asyncio.to_thread(fetch_reviews),
        asyncio.to_thread(fetch_resume),
    )

    records = records_resp.data or []
    reviews = reviews_resp.data or []
    resume = (resume_resp.data or [{}])[0].get("parsed_data") if resume_resp.data else None
    if isinstance(resume, str):
        try:
            resume = json.loads(resume)
        except json.JSONDecodeError:
            resume = None

    # ── review history, bucketed by local day ────────────────────────────────
    per_day = Counter()
    for r in reviews:
        d = _as_date(r.get("reviewed_at"))
        if d:
            per_day[d] += 1

    bars = []
    for i in range(BAR_DAYS - 1, -1, -1):
        d = today - timedelta(days=i)
        bars.append({"date": d.isoformat(), "n": per_day.get(d, 0)})

    this_week = sum(per_day.get(today - timedelta(days=i), 0) for i in range(7))
    last_week = sum(per_day.get(today - timedelta(days=i), 0) for i in range(7, 14))

    # A streak counts back from today, tolerating a review-free today so the
    # number does not collapse before the day's first card.
    streak = 0
    cursor = today if per_day.get(today) else today - timedelta(days=1)
    while per_day.get(cursor):
        streak += 1
        cursor -= timedelta(days=1)

    # ── collection state ─────────────────────────────────────────────────────
    due_count = 0
    strong = 0
    reviewed = 0
    for r in records:
        if _as_date(r.get("due_date")) and _as_date(r.get("due_date")) <= today:
            due_count += 1
        if r.get("review_count"):
            reviewed += 1
            if r.get("last_ease") in ("good", "easy"):
                strong += 1

    strong_pct = round(strong / reviewed * 100) if reviewed else 0

    # ── gap report: each resume skill against the answers behind it ──────────
    gap_rows = []
    if resume:
        skills = [s for s in (resume.get("skills") or []) if isinstance(s, str)]
        where = _skill_locations(resume)
        for skill in skills[:24]:
            count = _count_for_skill(skill, records)
            gap_rows.append(
                {
                    "skill": skill,
                    "where": where.get(skill.lower(), "listed on resume"),
                    "count": count,
                }
            )
        gap_rows.sort(key=lambda g: g["count"])

    # ── cards that keep slipping ─────────────────────────────────────────────
    weak = []
    for r in records:
        ease = r.get("last_ease")
        if ease == "again" or not r.get("review_count"):
            questions = r["questions_json"]
            if isinstance(questions, str):
                questions = json.loads(questions)
            weak.append(
                {
                    "id": r["id"],
                    "questions": questions,
                    "answer": r["answer"],
                    "tags": r.get("tags") or [],
                    "due_date": r.get("due_date"),
                    "review_count": r.get("review_count") or 0,
                    "last_ease": ease,
                    "ease_label": "Again" if ease == "again" else "Never reviewed",
                }
            )
    weak.sort(key=lambda w: (w["last_ease"] != "again", w["review_count"]))

    return {
        "bars": bars,
        "reviews_this_week": this_week,
        "reviews_last_week": last_week,
        "streak": streak,
        "due_count": due_count,
        "total": len(records),
        "strong_pct": strong_pct,
        "gap_rows": gap_rows,
        "untouched_skills": sum(1 for g in gap_rows if g["count"] == 0),
        "weak": weak[:8],
        "week_of": (today - timedelta(days=today.weekday())).isoformat(),
    }


def _as_date(value):
    if not value:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed.date()
    except ValueError:
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            return None


def _skill_locations(resume: dict) -> dict:
    """Where each skill shows up, so the gap report can say 'Nimbus, 2 projects'."""
    locations = {}
    for exp in resume.get("experience") or []:
        blob = json.dumps(exp).lower()
        company = exp.get("company") or ""
        for skill in resume.get("skills") or []:
            if isinstance(skill, str) and skill.lower() in blob and company:
                locations.setdefault(skill.lower(), company)
    for proj in resume.get("projects") or []:
        for tech in proj.get("technologies") or []:
            if isinstance(tech, str):
                locations.setdefault(tech.lower(), proj.get("name") or "a project")
    return locations


def _count_for_skill(skill: str, records: list) -> int:
    needle = skill.lower()
    count = 0
    for r in records:
        tags = [t.lower() for t in (r.get("tags") or [])]
        if needle in tags:
            count += 1
            continue
        questions = r["questions_json"]
        if isinstance(questions, str):
            questions = json.loads(questions)
        haystack = " ".join(questions).lower() + " " + (r.get("answer") or "").lower()
        if needle in haystack:
            count += 1
    return count
