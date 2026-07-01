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

app = FastAPI(title="Interview RAG API")

# Setup CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://localhost:3000", 
        "https://interview-rag-frontend.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(qa_router)
app.include_router(chat_router)
app.include_router(pdf_router)

logger.info("Interview RAG API has started successfully.")
