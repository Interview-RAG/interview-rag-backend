# InterviewRAG System

A smart, automated interview preparation application designed to collect, refine, and query daily interview questions and answers. It features an intelligent RAG (Retrieval-Augmented Generation) chatbot, dynamic deduplication, smart AI answer generation, and bulk PDF uploads.

## Features

- **Daily Q&A Collection:** Seamlessly add daily interview questions and answers manually.
- **Smart Generative Answers:** Don't know the answer? Just enter a question, leave the answer blank, and the AI will generate a comprehensive, accurate answer for you to review and save.
- **Bulk Q&A Upload via PDF:** Upload any PDF containing technical content or interview questions. The AI will intelligently extract, parse, and organize the text into distinct Q&A pairs for your collection.
- **Smart Deduplication & Refinement:** When new questions are added (either manually or via PDF), the system checks for similarities. If it's a match, it gracefully merges the questions and uses the Groq LLM (LLaMA-3) to refine and combine the answers.
- **Interactive RAG Chatbot (Agentic):** A conversational UI powered by LangGraph. The LLM acts as an autonomous agent that decides whether to answer normally, search your saved Q&A collection via Pinecone, or use DuckDuckGo to search the live web for current trends.
- **Persistent User Memory:** The chatbot remembers facts about you across sessions (stored in Supabase) to personalize your interview prep.
- **PDF Export:** Download your entire collection formatted clearly for offline study.
- **Premium UI:** Built with Vite and React, featuring a sleek dark-mode glassmorphism design.

## Architecture

- **Frontend:** React + Vite + Vanilla CSS
- **Backend:** Python FastAPI (Modular Microservice Architecture), LangChain & LangGraph
- **Database:** Supabase (Relational Storage & User Memory)
- **Vector Store:** Pinecone (Vector Similarity Search)
- **Embeddings:** HuggingFace `sentence-transformers` (all-MiniLM-L6-v2)
- **LLM Routing:** OpenRouter (Auto-routing to the best available models) with OpenAI-compatible tool calling.
- **Web Search:** DuckDuckGo Integration

## Prerequisites

- **Node.js** (v18+)
- **Python** (3.10+)
- **API Keys**: You will need active API keys for OpenRouter, Pinecone, and Supabase.

## Setup Instructions

1. **API Key Setup:**
   Navigate into the `backend` folder and create a `.env` file (you can copy `.env.example`). Add your keys:
   ```env
   OPENROUTER_API_KEY=your_actual_api_key_here
   PINECONE_API_KEY=your_pinecone_key
   # Add your supabase keys as required by your environment
   ```

2. **Quick Start (Windows):**
   Simply double-click the `run.bat` script located in the root directory. 
   This will automatically start both the backend API server and the React frontend in separate command windows.

3. **Manual Start:**
   - **Backend:**
     ```bash
     cd backend
     python -m venv venv
     .\venv\Scripts\activate
     pip install -r requirements.txt
     uvicorn main:app --reload
     ```
   - **Frontend:**
     ```bash
     cd frontend
     npm install
     npm run dev
     ```

## Usage

- Access the web interface at `http://localhost:5173`.
- Navigate using the sidebar:
  - **Collection:** View all your grouped Q&As.
  - **Add Q&A:** 
    - Input a new question and answer to save manually.
    - Leave the answer blank to have the AI generate one.
    - Use the **Bulk Upload via PDF** section to automatically parse Q&As from documents.
  - **Chatbot:** Ask anything related to your questions to get instant, context-aware answers.
  - **Export PDF:** Click to instantly download a `.pdf` of your collection.
