import json
from fastapi import APIRouter
from pydantic import BaseModel
import database
import rag

router = APIRouter(prefix="/api/chat", tags=["chat"])

class ChatQuery(BaseModel):
    query: str

@router.post("")
def chat_with_rag(query: ChatQuery):
    # 1. Embed query
    embedding = rag.get_embedding(query.query)
    
    # 2. Retrieve top-k documents from Pinecone
    try:
        results = rag.pinecone_index.query(
            vector=embedding,
            top_k=3
        )
    except Exception as e:
        print(f"Pinecone query error: {e}")
        results = None
    
    retrieved_docs = []
    if results and results.matches:
        for match in results.matches:
            resp = database.supabase.table("qa_records").select("*").eq("id", int(match.id)).execute()
            if resp.data:
                r = resp.data[0]
                qs = json.loads(r["questions_json"]) if isinstance(r["questions_json"], str) else r["questions_json"]
                retrieved_docs.append({
                    "questions": qs,
                    "answer": r["answer"]
                })
    
    # 3. Generate answer using Groq
    answer = rag.generate_rag_answer(query.query, retrieved_docs)
    
    return {"answer": answer}
