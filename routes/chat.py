import json
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import database
import rag
from rag_agent import app_graph
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

logger = logging.getLogger(__name__)

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
    
    session_title = None
    if len(resp.data) == 0:
        # First message in session, auto-generate title
        title = " ".join(query.query.split()[:5])
        if len(query.query.split()) > 5:
            title += "..."
        database.supabase.table("chat_sessions").update({"title": title}).eq("id", query.session_id).execute()
        session_title = title
    
    # 2. Convert to LangChain messages
    messages = []
    
    # Gemini has a 1 Million Token context window!
    for msg in resp.data:
        if not msg["content"].strip():
            continue
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
    logger.info(f"Invoking agent for session {query.session_id} with query: {query.query}")
    try:
        config = {"configurable": {"thread_id": query.session_id}}
        final_state = app_graph.invoke(
            {"messages": messages, "session_id": query.session_id},
            config=config
        )
        
        # 6. Check if graph is interrupted (HITL)
        state_snapshot = app_graph.get_state(config)
        if state_snapshot.next and "sensitive_tools" in state_snapshot.next:
            # We are interrupted before saving. Send ALL approval requests to frontend.
            last_message = final_state["messages"][-1]
            if hasattr(last_message, "tool_calls"):
                approvals_needed = []
                for tool_call in last_message.tool_calls:
                    if tool_call["name"] == "save_qa_to_collection":
                        args = tool_call["args"]
                        approvals_needed.append({
                            "question": args.get("question", ""), 
                            "answer": args.get("answer", ""), 
                            "tool_call_id": tool_call["id"]
                        })
                
                if approvals_needed:
                    logger.info(f"Graph interrupted. Sending {len(approvals_needed)} items for HITL approval.")
                    return {
                        "status": "requires_approval", 
                        "approvals": approvals_needed,
                        "answer_msg": f"I have drafted {len(approvals_needed)} Q&A pairs. Please review and approve them before I save.",
                        "session_title": session_title
                    }
        
        # 7. Normal completion
        final_message = final_state["messages"][-1]
        answer = final_message.content
        
        # 8. Save AI message to DB
        database.supabase.table("chat_messages").insert({
            "session_id": query.session_id,
            "role": "ai",
            "content": answer
        }).execute()
        
    except Exception as e:
        logger.error(f"Agent error in chat endpoint: {e}", exc_info=True)
        answer = "Sorry, I encountered an error answering your question."
    
    logger.info(f"Returning answer for session {query.session_id}")
    return {"status": "success", "answer": answer, "session_title": session_title}

# --- HITL Approve Save Endpoint ---
class ApprovalItem(BaseModel):
    tool_call_id: str
    question: str
    answer: str
    approved: bool

class ApproveSaveRequest(BaseModel):
    session_id: str
    approvals: list[ApprovalItem]

@router.post("/approve-save")
def approve_save(req: ApproveSaveRequest):
    config = {"configurable": {"thread_id": req.session_id}}
    state_snapshot = app_graph.get_state(config)
    
    if not state_snapshot.next or "sensitive_tools" not in state_snapshot.next:
        raise HTTPException(status_code=400, detail="No pending save approval found for this session.")
        
    from routes.qa import save_qa_logic
    
    # Process all approvals
    tool_messages = []
    
    # Map the approvals array for quick lookup
    approval_map = {item.tool_call_id: item for item in req.approvals}
    
    last_message = state_snapshot.values["messages"][-1]
    
    for tc in last_message.tool_calls:
        if tc["name"] == "save_qa_to_collection":
            item = approval_map.get(tc["id"])
            if item and item.approved:
                try:
                    result = save_qa_logic(item.question, item.answer)
                    msg = f"Successfully saved to collection: {result['message']}"
                    logger.info(f"User approved save for {tc['id']}")
                except Exception as e:
                    msg = f"Failed to save: {str(e)}"
                    logger.error(f"Failed to save during approval for {tc['id']}: {e}")
            else:
                msg = "User rejected the save request."
                logger.info(f"User rejected save for {tc['id']}")
                
            tool_messages.append(ToolMessage(tool_call_id=tc["id"], name=tc["name"], content=msg))

    # Update the graph state with ALL tool outputs, skipping the sensitive_tools node
    app_graph.update_state(
        config, 
        {"messages": tool_messages},
        as_node="sensitive_tools"
    )
    
    # Resume the graph
    final_state = app_graph.invoke(None, config=config)
    final_message = final_state["messages"][-1]
    
    # Save the final AI message to DB
    database.supabase.table("chat_messages").insert({
        "session_id": req.session_id,
        "role": "ai",
        "content": final_message.content
    }).execute()
    
    return {"status": "success", "answer": final_message.content}

