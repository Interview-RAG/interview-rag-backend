import os
import uuid
import json
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import database
import rag
import pdf_generator

app = FastAPI(title="Interview RAG API")

# Setup CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QACreate(BaseModel):
    question: str
    answer: str

class ChatQuery(BaseModel):
    query: str

@app.post("/api/qa")
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

@app.get("/api/qa")
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

@app.get("/api/export")
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

@app.post("/api/chat")
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
