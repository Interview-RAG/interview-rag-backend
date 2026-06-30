import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import database
import rag
from rag_agent import app_graph
from langchain_core.messages import HumanMessage, AIMessage

router = APIRouter(prefix="/api/chat", tags=["chat"])

# --- Session Management ---

class SessionCreate(BaseModel):
    title: str = "New Chat"

class SessionRename(BaseModel):
    title: str

@router.get("/sessions")
def get_sessions():
    resp = database.supabase.table("chat_sessions").select("*").order("created_at", desc=True).execute()
    return resp.data

@router.post("/sessions")
def create_session(session: SessionCreate):
    resp = database.supabase.table("chat_sessions").insert({"title": session.title}).execute()
    return resp.data[0]

@router.put("/sessions/{session_id}")
def rename_session(session_id: str, session: SessionRename):
    resp = database.supabase.table("chat_sessions").update({"title": session.title}).eq("id", session_id).execute()
    if not resp.data:
        raise HTTPException(status_code=404, detail="Session not found")
    return resp.data[0]

@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    database.supabase.table("chat_sessions").delete().eq("id", session_id).execute()
    return {"message": "Session deleted"}

@router.get("/sessions/{session_id}/messages")
def get_session_messages(session_id: str):
    resp = database.supabase.table("chat_messages").select("*").eq("session_id", session_id).order("created_at").execute()
    return resp.data

# --- Chat Endpoint ---

class ChatQuery(BaseModel):
    query: str
    session_id: str

@router.post("")
def chat_with_rag(query: ChatQuery):
    # 1. Fetch history from DB
    resp = database.supabase.table("chat_messages").select("*").eq("session_id", query.session_id).order("created_at").execute()
    
    # 2. Convert to LangChain messages
    messages = []
    for msg in resp.data:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            messages.append(AIMessage(content=msg["content"]))
            
    # 3. Add current query
    messages.append(HumanMessage(content=query.query))
    
    # 4. Save user message to DB
    database.supabase.table("chat_messages").insert({
        "session_id": query.session_id,
        "role": "user",
        "content": query.query
    }).execute()
    
    # 5. Invoke LangGraph
    try:
        final_state = app_graph.invoke(
            {"messages": messages, "session_id": query.session_id},
            config={"configurable": {"session_id": query.session_id}}
        )
        final_message = final_state["messages"][-1]
        answer = final_message.content
        
        # 6. Save AI message to DB
        database.supabase.table("chat_messages").insert({
            "session_id": query.session_id,
            "role": "ai",
            "content": answer
        }).execute()
        
    except Exception as e:
        print(f"Agent error: {e}")
        answer = "Sorry, I encountered an error answering your question."
    
    return {"answer": answer}

