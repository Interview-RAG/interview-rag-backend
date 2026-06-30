import json
import io
import time
import PyPDF2
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import database
import rag

router = APIRouter(prefix="/api/qa", tags=["qa"])

class QACreate(BaseModel):
    question: str
    answer: str

@router.post("")
def add_qa(qa: QACreate):
    # 1. Embed the incoming question
    embedding = rag.get_embedding(qa.question)
    
    # 2. Search for similar questions in Pinecone
    try:
        results = rag.pinecone_index.query(
            vector=embedding,
            top_k=1,
            include_metadata=True
        )
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
            resp = database.supabase.table("qa_records").select("*").eq("id", int(similar_id)).execute()
            
            if resp.data:
                existing_record = resp.data[0]
                questions = json.loads(existing_record['questions_json']) if isinstance(existing_record['questions_json'], str) else existing_record['questions_json']
                
                if qa.question not in questions:
                    questions.append(qa.question)
                
                # Combine answers using Groq LLM
                refined_answer = rag.combine_answers(qa.question, existing_record['answer'], qa.answer)
                
                # Update Supabase
                database.supabase.table("qa_records").update({
                    "questions_json": json.dumps(questions),
                    "answer": refined_answer
                }).eq("id", existing_record['id']).execute()
                
                return {"message": "Question was similar to an existing one. Grouped and refined answer.", "id": existing_record['id']}
    
    # If not similar, create a new record in Supabase
    new_record_resp = database.supabase.table("qa_records").insert({
        "questions_json": json.dumps([qa.question]),
        "answer": qa.answer
    }).execute()
    
    new_id = new_record_resp.data[0]['id']
    
    # Add to Pinecone for future similarity matching
    rag.pinecone_index.upsert(
        vectors=[{
            "id": str(new_id),
            "values": embedding,
            "metadata": {"answer": qa.answer}
        }]
    )
    
    return {"message": "New Q&A added successfully.", "id": new_id}

@router.get("")
def get_collection():
    resp = database.supabase.table("qa_records").select("*").execute()
    records = []
    for r in resp.data:
        qs = json.loads(r["questions_json"]) if isinstance(r["questions_json"], str) else r["questions_json"]
        records.append({
            "id": r["id"],
            "questions": qs,
            "answer": r["answer"]
        })
    return records


class QuestionQuery(BaseModel):
    question: str

@router.post("/generate-answer")
def generate_answer(query: QuestionQuery):
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    answer = rag.generate_answer_for_question(query.question)
    return {"answer": answer}

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
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
            
        # Parse text using Groq
        qa_pairs = rag.parse_pdf_text_to_qa(text)
        
        if not qa_pairs:
            return {"message": "No Q&A pairs found in the document.", "added": 0}
            
        added_count = 0
        for pair in qa_pairs:
            q = pair.get("question")
            a = pair.get("answer")
            if q and a:
                # Reuse the exact same duplicate/similarity logic from add_qa
                add_qa(QACreate(question=q, answer=a))
                added_count += 1
                time.sleep(1.5)  # Add a slight delay to prevent HuggingFace API rate limits / connection drops
                
        return {"message": f"Successfully extracted and saved {added_count} Q&A pairs.", "added": added_count}
        
    except Exception as e:
        print(f"PDF Upload Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
