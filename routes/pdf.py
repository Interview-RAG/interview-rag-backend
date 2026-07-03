import json
import uuid
import asyncio
from fastapi import APIRouter
from fastapi.responses import FileResponse
import database
import pdf_generator

router = APIRouter(prefix="/api/export", tags=["export"])

@router.get("")
async def export_pdf():
    def fetch_records():
        return database.supabase.table("qa_records").select("*").execute()
    resp = await asyncio.to_thread(fetch_records)
    qa_list = []
    for r in resp.data:
        qs = json.loads(r["questions_json"]) if isinstance(r["questions_json"], str) else r["questions_json"]
        qa_list.append({
            "questions": qs,
            "answer": r["answer"]
        })
    
    pdf_path = f"collection_{uuid.uuid4().hex[:8]}.pdf"
    def gen_pdf():
        pdf_generator.generate_pdf(qa_list, output_path=pdf_path)
    await asyncio.to_thread(gen_pdf)
    
    return FileResponse(
        path=pdf_path, 
        filename="Interview_Collection.pdf", 
        media_type="application/pdf",
        background=None
    )
