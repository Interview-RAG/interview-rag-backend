import json
import io
import asyncio
import PyPDF2
import base64
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import database
import rag

router = APIRouter(prefix="/api/qa", tags=["qa"])

class QACreate(BaseModel):
    question: str
    answer: str

async def save_qa_logic(question: str, answer: str):
    # 1. Embed the incoming question
    try:
        embedding = await rag.get_embedding(question)
    except Exception as e:
        print(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail="Failed to embed question.")
        
    # 2. Search for similar questions in Pinecone
    try:
        def query_pinecone():
            return rag.pinecone_index.query(vector=embedding, top_k=1, include_metadata=True)
        results = await asyncio.to_thread(query_pinecone)
    except Exception as e:
        print(f"Pinecone search error: {e}")
        results = None

    # Pinecone Cosine Similarity threshold (higher is more similar, > 0.8 is generally very similar)
    SIMILARITY_THRESHOLD = 0.85 
    
    if results and results.matches:
        match = results.matches[0]
        if match.score > SIMILARITY_THRESHOLD:
            similar_id = match.id
            
            # Find the existing record in Supabase
            def query_supabase():
                return database.supabase.table("qa_records").select("*").eq("id", int(similar_id)).execute()
            resp = await asyncio.to_thread(query_supabase)
            
            if resp.data:
                existing_record = resp.data[0]
                questions = json.loads(existing_record['questions_json']) if isinstance(existing_record['questions_json'], str) else existing_record['questions_json']
                
                if question not in questions:
                    questions.append(question)
                
                # Combine answers using Groq LLM
                refined_answer = await rag.combine_answers(question, existing_record['answer'], answer)
                
                # Update Supabase
                def update_supabase():
                    return database.supabase.table("qa_records").update({
                        "questions_json": json.dumps(questions),
                        "answer": refined_answer
                    }).eq("id", existing_record['id']).execute()
                await asyncio.to_thread(update_supabase)
                
                return {"message": "Question was similar to an existing one. Grouped and refined answer.", "id": existing_record['id']}
    
    # If not similar, create a new record in Supabase
    def insert_supabase():
        return database.supabase.table("qa_records").insert({
            "questions_json": json.dumps([question]),
            "answer": answer
        }).execute()
    new_record_resp = await asyncio.to_thread(insert_supabase)
    
    new_id = new_record_resp.data[0]['id']
    
    # Add to Pinecone for future similarity matching
    def upsert_pinecone():
        rag.pinecone_index.upsert(
            vectors=[{
                "id": str(new_id),
                "values": embedding,
                "metadata": {"answer": answer}
            }]
        )
    await asyncio.to_thread(upsert_pinecone)
    
    return {"message": "New Q&A added successfully.", "id": new_id}

@router.post("")
async def add_qa(qa: QACreate):
    is_valid = await rag.is_interview_related(qa.question + " " + qa.answer)
    if not is_valid:
        raise HTTPException(status_code=400, detail="This system only accepts interview-related questions and answers.")
    return await save_qa_logic(qa.question, qa.answer)

@router.get("")
async def get_collection():
    def get_records():
        return database.supabase.table("qa_records").select("*").execute()
    resp = await asyncio.to_thread(get_records)
    records = []
    for r in resp.data:
        qs = json.loads(r["questions_json"]) if isinstance(r["questions_json"], str) else r["questions_json"]
        records.append({
            "id": r["id"],
            "questions": qs,
            "answer": r["answer"]
        })
    return records

@router.delete("/{qa_id}")
async def delete_qa(qa_id: int):
    # Delete from Supabase first
    def delete_supabase():
        return database.supabase.table("qa_records").delete().eq("id", qa_id).execute()
    resp = await asyncio.to_thread(delete_supabase)
    
    if not resp.data:
        raise HTTPException(status_code=404, detail="Q&A not found")
        
    # Delete from Pinecone
    try:
        def delete_pinecone():
            rag.pinecone_index.delete(ids=[str(qa_id)])
        await asyncio.to_thread(delete_pinecone)
    except Exception as e:
        print(f"Pinecone delete error: {e}")
        
    return {"message": "Deleted successfully"}


class QuestionQuery(BaseModel):
    question: str

@router.post("/generate-answer")
async def generate_answer(query: QuestionQuery):
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
        
    is_valid = await rag.is_interview_related(query.question)
    if not is_valid:
        raise HTTPException(status_code=400, detail="This system only answers interview-related questions.")
        
    answer = await rag.generate_answer_for_question(query.question)
    return {"answer": answer}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename.lower()
    
    if filename.endswith('.pdf'):
        try:
            content = await file.read()
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
                
            if not text.strip():
                raise HTTPException(status_code=400, detail="Could not extract text from PDF")
                
            qa_pairs = await rag.parse_pdf_text_to_qa(text)
        except Exception as e:
            print(f"PDF Parsing Error: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
            
    elif filename.endswith(('.png', '.jpeg', '.jpg', '.webp')):
        try:
            content = await file.read()
            base64_image = base64.b64encode(content).decode('utf-8')
            mime_type = file.content_type
            qa_pairs = await rag.parse_image_to_qa(base64_image, mime_type)
        except Exception as e:
            print(f"Image Parsing Error: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing Image: {str(e)}")
            
    else:
        raise HTTPException(status_code=400, detail="Only PDF and Image files (PNG, JPEG, WebP) are supported")
    
    try:
        if not qa_pairs:
            return {"message": "No Q&A pairs found in the document.", "added": 0}
            
        added_count = 0
        for pair in qa_pairs:
            q = pair.get("question")
            a = pair.get("answer")
            if q and a:
                # We do not need to re-validate here because the LLM prompt already enforces it
                await save_qa_logic(q, a)
                added_count += 1
                await asyncio.sleep(1.5)  # Add a slight delay to prevent HuggingFace API rate limits / connection drops
                
        return {"message": f"Successfully extracted and saved {added_count} Q&A pairs.", "added": added_count}
        
    except Exception as e:
        print(f"Upload Save Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving extracted Q&A: {str(e)}")
