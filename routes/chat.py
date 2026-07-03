import json
import logging
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
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
async def get_sessions():
    def fetch_sessions():
        return database.supabase.table("chat_sessions").select("*").order("created_at", desc=True).execute()
    resp = await asyncio.to_thread(fetch_sessions)
    return resp.data

@router.post("/sessions")
async def create_session(session: SessionCreate):
    def insert_session():
        return database.supabase.table("chat_sessions").insert({"title": session.title}).execute()
    resp = await asyncio.to_thread(insert_session)
    return resp.data[0]

@router.put("/sessions/{session_id}")
async def rename_session(session_id: str, session: SessionRename):
    def update_session():
        return database.supabase.table("chat_sessions").update({"title": session.title}).eq("id", session_id).execute()
    resp = await asyncio.to_thread(update_session)
    if not resp.data:
        raise HTTPException(status_code=404, detail="Session not found")
    return resp.data[0]

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    def run_delete():
        return database.supabase.table("chat_sessions").delete().eq("id", session_id).execute()
    await asyncio.to_thread(run_delete)
    return {"message": "Session deleted"}

@router.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    def fetch_messages():
        return database.supabase.table("chat_messages").select("*").eq("session_id", session_id).order("created_at").execute()
    resp = await asyncio.to_thread(fetch_messages)
    return resp.data

# --- Chat Endpoint ---

class ChatQuery(BaseModel):
    query: str
    session_id: str

@router.post("")
async def chat_with_rag(query: ChatQuery):
    # 1. Fetch history from DB
    def fetch_msgs():
        return database.supabase.table("chat_messages").select("*").eq("session_id", query.session_id).order("created_at").execute()
    resp = await asyncio.to_thread(fetch_msgs)
    
    session_title = None
    if len(resp.data) == 0:
        # First message in session, auto-generate title
        title = " ".join(query.query.split()[:5])
        if len(query.query.split()) > 5:
            title += "..."
        def update_title():
            return database.supabase.table("chat_sessions").update({"title": title}).eq("id", query.session_id).execute()
        await asyncio.to_thread(update_title)
        session_title = title
    
    # 2. Convert to LangChain messages (Limit to last 10 to avoid Groq 8k TPM limits)
    messages = []
    
    # Slice to keep only the last 10 messages from the history
    recent_history = resp.data[-10:] if len(resp.data) > 10 else resp.data
    
    for msg in recent_history:
        if not msg["content"].strip():
            continue
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            messages.append(AIMessage(content=msg["content"]))
            
    # 3. Add current query
    messages.append(HumanMessage(content=query.query))
    
    # 4. Save user message to DB
    def insert_user_msg():
        return database.supabase.table("chat_messages").insert({
            "session_id": query.session_id,
            "role": "user",
            "content": query.query
        }).execute()
    await asyncio.to_thread(insert_user_msg)
    
    # 5. Invoke LangGraph via Streaming
    logger.info(f"Streaming agent for session {query.session_id} with query: {query.query}")
    
    async def event_stream():
        config = {"configurable": {"thread_id": query.session_id}}
        
        if session_title:
            yield f"data: {json.dumps({'type': 'session_title', 'title': session_title})}\n\n"
            
        try:
            # Use astream with stream_mode="updates" instead of astream_events.
            # astream_events forces Groq into streaming mode, which causes Llama 3.3
            # to malform tool call names (e.g. 'search_web {"query": "..."}').
            # stream_mode="updates" uses non-streaming invoke internally per node,
            # then emits the result of each node as it completes — giving us real-time
            # tool notifications without triggering Groq's streaming tool call bug.
            async for chunk in app_graph.astream(
                {"messages": messages, "session_id": query.session_id},
                config=config,
                stream_mode="updates"
            ):
                for node_name, state_update in chunk.items():
                    # When the "agent" node finishes, check if it decided to use tools
                    if node_name == "agent":
                        last_msg = state_update["messages"][-1]
                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                yield f"data: {json.dumps({'type': 'tool_start', 'name': tc['name']})}\n\n"
                    
                    # When a "tools" node finishes, the tool has completed execution
                    elif node_name == "tools":
                        yield f"data: {json.dumps({'type': 'tool_end', 'name': 'tool'})}\n\n"
            
            # 6. Check if graph is interrupted (HITL)
            state_snapshot = app_graph.get_state(config)
            if state_snapshot.next and "sensitive_tools" in state_snapshot.next:
                last_message = state_snapshot.values["messages"][-1]
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
                        yield f"data: {json.dumps({'type': 'requires_approval', 'approvals': approvals_needed, 'answer_msg': f'I have drafted {len(approvals_needed)} Q&A pairs. Please review and approve them before I save.'})}\n\n"
                        return
                        
            # 7. Normal completion
            final_message = state_snapshot.values["messages"][-1]
            answer = final_message.content
            
            # 8. Save AI message to DB
            def insert_ai_msg():
                return database.supabase.table("chat_messages").insert({
                    "session_id": query.session_id,
                    "role": "ai",
                    "content": answer
                }).execute()
            await asyncio.to_thread(insert_ai_msg)
            
            yield f"data: {json.dumps({'type': 'final_answer', 'content': answer})}\n\n"
            
        except Exception as e:
            logger.error(f"Agent error in stream endpoint: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': 'Sorry, I encountered an error answering your question.'})}\n\n"
            
    return StreamingResponse(event_stream(), media_type="text/event-stream")

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
async def approve_save(req: ApproveSaveRequest):
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
                    result = await save_qa_logic(item.question, item.answer)
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
    final_state = await app_graph.ainvoke(None, config=config)
    final_message = final_state["messages"][-1]
    
    # Save the final AI message to DB
    def insert_final_msg():
        return database.supabase.table("chat_messages").insert({
            "session_id": req.session_id,
            "role": "ai",
            "content": final_message.content
        }).execute()
    await asyncio.to_thread(insert_final_msg)
    
    return {"status": "success", "answer": final_message.content}

