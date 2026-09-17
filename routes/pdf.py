import json
import os
import uuid
import asyncio
from collections import defaultdict

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from auth import get_current_user
import database
import pdf_generator

router = APIRouter(prefix="/api/export", tags=["export"])

UNTAGGED = "Untagged"


@router.get("")
async def export_pdf(user_id: str = Depends(get_current_user)):
    """The study guide: the caller's own collection, grouped by tag."""
    def fetch_records():
        return (
            database.supabase.table("qa_records")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )

    resp = await asyncio.to_thread(fetch_records)

    # A pair with several tags appears under each of them, which is what you
    # want in a study guide you flip through by topic.
    grouped = defaultdict(list)
    for r in resp.data or []:
        qs = r["questions_json"]
        if isinstance(qs, str):
            qs = json.loads(qs)
        entry = {"questions": qs, "answer": r["answer"]}
        tags = r.get("tags") or []
        for tag in tags or [UNTAGGED]:
            grouped[tag].append(entry)

    # Untagged last, everything else alphabetical.
    sections = [
        {"tag": tag, "items": grouped[tag]}
        for tag in sorted(grouped, key=lambda t: (t == UNTAGGED, t.lower()))
    ]

    pdf_path = f"collection_{uuid.uuid4().hex[:8]}.pdf"

    def gen_pdf():
        pdf_generator.generate_pdf(sections, output_path=pdf_path)

    await asyncio.to_thread(gen_pdf)

    def cleanup():
        try:
            os.remove(pdf_path)
        except OSError:
            pass

    return FileResponse(
        path=pdf_path,
        filename="PrepAI_Study_Guide.pdf",
        media_type="application/pdf",
        background=BackgroundTask(cleanup),
    )
