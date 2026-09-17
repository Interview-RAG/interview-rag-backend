import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

from routes.qa import router as qa_router
from routes.chat import router as chat_router
from routes.pdf import router as pdf_router
from routes.user import router as user_router
from routes.auth_routes import router as auth_router
from routes.resume import router as resume_router
from routes.practice import router as practice_router
from routes.progress import router as progress_router

app = FastAPI(title="PrepAI API")

# Setup CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://interview-rag-frontend.vercel.app"
    ],
    # Vite falls back to 5174+ when 5173 is busy, and `vite preview` uses 4173.
    # Without this the browser can't read error bodies and every failure shows
    # as a generic toast.
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(qa_router)
app.include_router(chat_router)
app.include_router(pdf_router)
app.include_router(user_router)
app.include_router(auth_router)
app.include_router(resume_router)
app.include_router(practice_router)
app.include_router(progress_router)

logger.info("PrepAI API has started successfully.")
