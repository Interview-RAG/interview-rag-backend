import os
import json
import io
import asyncio
import PyPDF2
import fitz  # PyMuPDF for image extraction
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from auth import get_current_user
import database
import logging
from rag import parse_resume_text, is_resume

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/resume", tags=["resume"])

async def extract_profile_pic_from_pdf(file_bytes, user_id):
    """
    Scans the first page of the PDF for images.
    Uses heuristics to determine if an image is a profile picture (e.g. square-ish, not an icon, not a background).
    Uploads the best match to Supabase Storage.
    """
    try:
        def process():
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            if doc.page_count == 0:
                return None
            
            page = doc[0]
            image_list = page.get_images(full=True)
            if not image_list:
                return None
                
            best_image = None
            best_score = 0
            
            for img in image_list:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                width = base_image["width"]
                height = base_image["height"]
                ext = base_image["ext"]
                
                # Heuristics: width/height > 80 (ignore icons) and < 1500 (ignore page backgrounds)
                if 80 < width < 1500 and 80 < height < 1500:
                    aspect_ratio = width / height
                    # Profile pics are usually close to square
                    if 0.5 < aspect_ratio < 2.0:
                        score = width * height
                        # Pick the largest image that meets the criteria
                        if score > best_score:
                            best_score = score
                            best_image = {
                                "bytes": image_bytes,
                                "ext": ext
                            }
                            
            if best_image:
                filename = f"{user_id}/profile.{best_image['ext']}"
                # Upload to Supabase Storage bucket
                database.supabase.storage.from_("resume_images").upload(
                    path=filename,
                    file=best_image["bytes"],
                    file_options={"content-type": f"image/{best_image['ext']}", "upsert": "true"}
                )
                public_url = database.supabase.storage.from_("resume_images").get_public_url(filename)
                return public_url
                
            return None
            
        return await asyncio.to_thread(process)
    except Exception as e:
        logger.error(f"Failed to extract profile picture: {e}")
        return None


@router.post("/upload")
async def upload_resume(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user)
):
    """
    Accepts a PDF resume, parses it using LLM, and stores structured data in Supabase.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    try:
        # Read the file bytes completely once
        file_bytes = await file.read()
        
        # 1. Read PDF text
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
                
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
            
        # 2. Validate if it's a resume
        logger.info(f"Validating document type for user {user_id}...")
        valid_resume = await is_resume(text)
        if not valid_resume:
            raise HTTPException(status_code=400, detail="The uploaded document does not appear to be a resume or CV. Please upload a valid resume.")
            
        # 3. Parse using LLM
        logger.info(f"Parsing resume for user {user_id}...")
        parsed_data = await parse_resume_text(text)
        
        if not parsed_data:
            raise HTTPException(status_code=500, detail="Failed to parse resume content.")
            
        # 4. Attempt to extract profile picture
        logger.info(f"Attempting to extract profile picture for user {user_id}...")
        profile_url = await extract_profile_pic_from_pdf(file_bytes, user_id)
        if profile_url:
            parsed_data["profile_picture_url"] = profile_url
            
        # Calculate General ATS Score
        from ats_scorer import calculate_general_score
        general_score_data = calculate_general_score(parsed_data)
        parsed_data["ats_general_score"] = general_score_data
        
        # 5. Store in Supabase
        def save_to_db():
            # Upsert or Insert (Since user_id is the logical key, we can check if it exists)
            # Check if exists
            existing = database.supabase.table("user_resumes").select("id").eq("user_id", user_id).execute()
            
            data = {
                "user_id": user_id,
                "raw_text": text,
                "parsed_data": parsed_data
            }
            
            if existing.data and len(existing.data) > 0:
                # Update
                return database.supabase.table("user_resumes").update(data).eq("user_id", user_id).execute()
            else:
                # Insert
                return database.supabase.table("user_resumes").insert(data).execute()
                
        await asyncio.to_thread(save_to_db)
        
        return {"status": "success", "parsed_data": parsed_data}
        
    except Exception as e:
        logger.error(f"Error processing resume upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("")
async def get_resume(user_id: str = Depends(get_current_user)):
    """
    Fetches the current user's parsed resume data.
    """
    try:
        def fetch_db():
            return database.supabase.table("user_resumes").select("*").eq("user_id", user_id).execute()
            
        result = await asyncio.to_thread(fetch_db)
        
        if not result.data or len(result.data) == 0:
            return {"status": "success", "resume": None}
            
        return {"status": "success", "resume": result.data[0]}
        
    except Exception as e:
        logger.error(f"Error fetching resume: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("")
async def delete_resume(user_id: str = Depends(get_current_user)):
    """
    Deletes the current user's resume from the database and removes the profile picture from storage.
    """
    try:
        def process_delete():
            # 1. Delete from DB
            database.supabase.table("user_resumes").delete().eq("user_id", user_id).execute()
            
            # 2. Delete from Storage
            # Supabase Storage list() requires a path. It returns a list of dictionaries if files exist.
            try:
                files = database.supabase.storage.from_("resume_images").list(user_id)
                if files and isinstance(files, list):
                    file_paths = [f"{user_id}/{f['name']}" for f in files if f.get('name')]
                    if file_paths:
                        database.supabase.storage.from_("resume_images").remove(file_paths)
            except Exception as storage_err:
                logger.warning(f"Failed to delete images from storage (might not exist): {storage_err}")
                
        await asyncio.to_thread(process_delete)
        return {"status": "success", "message": "Resume deleted successfully."}
        
    except Exception as e:
        logger.error(f"Error deleting resume: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class MatchRequest(BaseModel):
    job_description: str

@router.post("/match")
async def match_resume(
    request: MatchRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Calculates a targeted ATS score based on a provided job description.
    """
    try:
        from rag import extract_jd_keywords
        jd_keywords = await extract_jd_keywords(request.job_description)
        
        def process_match():
            result = database.supabase.table("user_resumes").select("parsed_data").eq("user_id", user_id).execute()
            if not result.data or len(result.data) == 0:
                raise HTTPException(status_code=404, detail="No resume found. Please upload one first.")
                
            parsed_data = result.data[0]["parsed_data"]
            
            from ats_scorer import calculate_targeted_score
            targeted_score_data = calculate_targeted_score(parsed_data, jd_keywords)
            
            # Save the score and JD in the parsed data so it persists
            parsed_data["ats_targeted_score"] = targeted_score_data
            parsed_data["targeted_job_description"] = request.job_description
            
            database.supabase.table("user_resumes").update({"parsed_data": parsed_data}).eq("user_id", user_id).execute()
            
            return targeted_score_data
            
        score_data = await asyncio.to_thread(process_match)
        return {"status": "success", "score_data": score_data}
        
    except Exception as e:
        logger.error(f"Error in targeted match: {e}")
        raise HTTPException(status_code=500, detail=str(e))
