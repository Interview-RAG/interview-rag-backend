import json
import uuid
from fastapi import APIRouter
from fastapi.responses import FileResponse
import database
import pdf_generator

router = APIRouter(prefix="/api/export", tags=["export"])

@router.get("")
def export_pdf():
    resp = database.supabase.table("qa_records").select("*").execute()
    qa_list = []
    for r in resp.data:
        qs = json.loads(r["questions_json"]) if isinstance(r["questions_json"], str) else r["questions_json"]
        qa_list.append({
            "questions": qs,
            "answer": r["answer"]
        })
    
    pdf_path = f"collection_{uuid.uuid4().hex[:8]}.pdf"
    pdf_generator.generate_pdf(qa_list, output_path=pdf_path)
    
    return FileResponse(
        path=pdf_path, 
        filename="Interview_Collection.pdf", 
        media_type="application/pdf",
        background=None
    )
