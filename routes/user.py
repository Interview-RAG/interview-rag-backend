import asyncio
from fastapi import APIRouter, HTTPException, Depends
from auth import get_current_user
import database
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/user", tags=["user"])

@router.delete("")
async def delete_account(user_id: str = Depends(get_current_user)):
    """
    Deletes the current user's account from the custom users table and all associated data.
    Requires an authenticated user token.
    """
    try:
        # Delete user's QA records from Supabase
        def delete_qa():
            return database.supabase.table("qa_records").delete().eq("user_id", user_id).execute()
        await asyncio.to_thread(delete_qa)

        # Delete user's facts from Supabase
        def delete_facts():
            return database.supabase.table("user_facts").delete().eq("user_id", user_id).execute()
        await asyncio.to_thread(delete_facts)

        # Delete the user record itself
        def delete_user():
            return database.supabase.table("users").delete().eq("id", user_id).execute()
        await asyncio.to_thread(delete_user)

        logger.info(f"User {user_id} successfully deleted their account.")
        return {"status": "success", "message": "Account deleted successfully."}
    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete account: {str(e)}")
