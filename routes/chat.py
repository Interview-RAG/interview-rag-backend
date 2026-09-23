import json
import logging
import asyncio
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from auth import get_current_user
import database
import rag
import llm
from rag_agent import app_graph
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# --- Session Management ---

class SessionCreate(BaseModel):
    title: str = "New Chat"

class SessionRename(BaseModel):
    title: str

GENERIC_ERROR = "Sorry, I encountered an error answering your question."
PROVIDER_CREDITS_ERROR = (
    "The AI provider has run out of credits, so the coach can't reply right now. "
    "This needs the administrator to top up the provider account — nothing on your side."
)
PROVIDER_BUSY_ERROR = (
    "The AI coach is temporarily over its free usage limits. "
    "Please try again in a minute."
)


def user_facing_error(exc: Exception) -> str:
    """Turn an agent failure into something the chat bubble can show honestly.

    A 402 from a provider is not a transient glitch: the account has no credits
    and *every* turn will fail until someone adds some. Saying so beats a
    generic apology that leaves the user retrying and the operator guessing —
    every turn 402'd for days behind the same "Sorry, I encountered an error"
    text before this was noticed (2026-09-23).

    LLMUnavailable means every configured provider is rate-limited or cooling
    down right now; that one *is* transient, so say so.
    """
    if isinstance(exc, llm.LLMUnavailable):
        return PROVIDER_BUSY_ERROR
    if getattr(exc, "status_code", None) == 402 or "Error code: 402" in str(exc):
        return PROVIDER_CREDITS_ERROR
    return GENERIC_ERROR


@router.get("/sessions")
async def get_sessions(user_id: str = Depends(get_current_user)):
    """Sessions newest-activity first, each with last_message_at and message_count.

    Ordering by created_at alone reopened a two-month-old thread on load (the
    user typed "Hi" and got "Hi again! I've got the recommendation engine
    answer ready…"), and two sessions created in the same second tied, so the
    pick was random. Sort by when the conversation was last *used*, and break
    ties on id so the order is stable across requests.
    """
    def fetch_sessions():
        return database.supabase.table("chat_sessions").select("*").eq("user_id", user_id).execute()

    def fetch_activity():
        return (database.supabase.table("chat_messages")
                .select("session_id, created_at").eq("user_id", user_id).execute())

    sessions_resp, activity_resp = await asyncio.gather(
        asyncio.to_thread(fetch_sessions), asyncio.to_thread(fetch_activity)
    )

    last_at, count = {}, {}
    for m in activity_resp.data or []:
        sid = m["session_id"]
        count[sid] = count.get(sid, 0) + 1
        if m["created_at"] > last_at.get(sid, ""):
            last_at[sid] = m["created_at"]

    sessions = []
    for s in sessions_resp.data or []:
        s = dict(s)
        s["last_message_at"] = last_at.get(s["id"])
        s["message_count"] = count.get(s["id"], 0)
        sessions.append(s)

    # ISO-8601 strings from Postgres compare correctly as text.
    sessions.sort(key=lambda s: (s["last_message_at"] or s["created_at"], str(s["id"])), reverse=True)
    return sessions

@router.post("/sessions")
async def create_session(session: SessionCreate, user_id: str = Depends(get_current_user)):
    def insert_session():
        return database.supabase.table("chat_sessions").insert({"title": session.title, "user_id": user_id}).execute()
    resp = await asyncio.to_thread(insert_session)
    return resp.data[0]

@router.put("/sessions/{session_id}")
async def rename_session(session_id: str, session: SessionRename, user_id: str = Depends(get_current_user)):
    def update_session():
        return database.supabase.table("chat_sessions").update({"title": session.title}).eq("id", session_id).eq("user_id", user_id).execute()
    resp = await asyncio.to_thread(update_session)
    if not resp.data:
        raise HTTPException(status_code=404, detail="Session not found")
    return resp.data[0]

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user_id: str = Depends(get_current_user)):
    def run_delete():
        return database.supabase.table("chat_sessions").delete().eq("id", session_id).eq("user_id", user_id).execute()
    await asyncio.to_thread(run_delete)
    return {"message": "Session deleted"}

@router.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, user_id: str = Depends(get_current_user)):
    def fetch_messages():
        return database.supabase.table("chat_messages").select("*").eq("session_id", session_id).eq("user_id", user_id).order("created_at").execute()
    resp = await asyncio.to_thread(fetch_messages)
    return resp.data

# --- Chat Endpoint ---

class ChatQuery(BaseModel):
    query: str
    session_id: str
    # Coach answers and critiques; mock interview asks one question at a time;
    # pressure test interrogates a project from the resume.
    mode: str = "coach"
    project: str | None = None

@router.post("")
async def chat_with_rag(query: ChatQuery, user_id: str = Depends(get_current_user)):
    # Verify session ownership first
    def check_session():
        return database.supabase.table("chat_sessions").select("id").eq("id", query.session_id).eq("user_id", user_id).execute()
    session_check = await asyncio.to_thread(check_session)
    if not session_check.data:
        raise HTTPException(status_code=403, detail="Not authorized to access this session")

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
    
    # 2. Convert to LangChain messages. Only the last 10 go to the model: the
    # agent tier can fall back to Groq, whose per-minute token budget is 8K.
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
            "content": query.query,
            "user_id": user_id
        }).execute()
    inserted = await asyncio.to_thread(insert_user_msg)
    # Kept so a failed turn can remove its own row again (see the except below).
    user_msg_id = inserted.data[0]["id"] if inserted.data else None

    # 5. Invoke LangGraph via Streaming
    logger.info(f"Streaming agent for session {query.session_id} with query: {query.query}")
    
    async def event_stream():
        mode = query.mode if query.mode in ("coach", "mock", "pressure", "jobs") else "coach"
        config = {
            "configurable": {
                "thread_id": query.session_id,
                "user_id": user_id,
                "mode": mode,
                "project": query.project
            },
            # LangGraph defaults to 25, which is ~12 agent/tool round trips —
            # enough for a model to thrash on an unsatisfiable search and burn
            # a dozen LLM calls. Six round trips is plenty for any real answer.
            "recursion_limit": 12,
        }
        
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
            #
            # The model often writes prose in the same turn it calls a tool, then
            # more prose after the result comes back. Taking only messages[-1]
            # silently dropped everything but the last chunk — a job search could
            # return "here is how these fit you" with the actual listings gone.
            # Gather every assistant message since the user's turn instead.
            history = state_snapshot.values["messages"]
            parts = []
            for msg in reversed(history):
                if isinstance(msg, HumanMessage):
                    break
                if isinstance(msg, AIMessage):
                    text = msg.content.strip() if isinstance(msg.content, str) else ""
                    if text and text not in parts:
                        parts.append(text)
            answer = "\n\n".join(reversed(parts))
            if not answer:
                last = history[-1].content if history else ""
                answer = last if isinstance(last, str) else ""
            
            # 8. Save AI message to DB
            def insert_ai_msg():
                return database.supabase.table("chat_messages").insert({
                    "session_id": query.session_id,
                    "role": "ai",
                    "content": answer,
                    "user_id": user_id
                }).execute()
            await asyncio.to_thread(insert_ai_msg)
            
            yield f"data: {json.dumps({'type': 'final_answer', 'content': answer})}\n\n"
            
        except Exception as e:
            logger.error(f"Agent error in stream endpoint: {e}", exc_info=True)
            # The user row was written in step 4, before streaming. With no reply
            # row it is an orphan, and the next turn's history then has two
            # consecutive user messages — the model answers both at once
            # (TECHNICAL_DEBT #15, observed 2026-09-17). Remove it so a failed
            # turn leaves no trace; the client still shows the error bubble.
            if user_msg_id is not None:
                def delete_orphan():
                    return (database.supabase.table("chat_messages").delete()
                            .eq("id", user_msg_id).eq("user_id", user_id).execute())
                try:
                    await asyncio.to_thread(delete_orphan)
                except Exception as cleanup_err:
                    logger.error(f"Could not remove orphan user message {user_msg_id}: {cleanup_err}")
            yield f"data: {json.dumps({'type': 'error', 'message': user_facing_error(e)})}\n\n"
            
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
async def approve_save(req: ApproveSaveRequest, user_id: str = Depends(get_current_user)):
    config = {"configurable": {"thread_id": req.session_id, "user_id": user_id}}
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
                    result = await save_qa_logic(item.question, item.answer, user_id)
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
            "content": final_message.content,
            "user_id": user_id
        }).execute()
    await asyncio.to_thread(insert_final_msg)
    
    return {"status": "success", "answer": final_message.content}

