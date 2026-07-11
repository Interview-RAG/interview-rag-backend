import os
import sys
import asyncio
import PyPDF2

sys.stdout.reconfigure(encoding='utf-8')
# Add backend directory to sys path so we can import modules properly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import rag
from routes.qa import save_qa_logic

async def ingest_pdf(filepath):
    print(f"\n--- Processing PDF: {os.path.basename(filepath)} ---")
    try:
        with open(filepath, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for i, page in enumerate(pdf_reader.pages):
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"Failed to read PDF {filepath}: {e}")
        return

    if not text.strip():
        print(f"Could not extract any text from {filepath}")
        return

    print("Parsing text to Q&A pairs via Groq (this may take a minute for large files)...")
    qa_pairs = await rag.parse_pdf_text_to_qa(text)
    
    if not qa_pairs:
        print("No Q&A pairs extracted.")
        return
        
    print(f"Extracted {len(qa_pairs)} Q&A pairs. Saving to Supabase & Pinecone...")
    
    added = 0
    for i, pair in enumerate(qa_pairs):
        q = pair.get("question")
        a = pair.get("answer")
        if q and a:
            try:
                await save_qa_logic(q, a)
                added += 1
                print(f"[{i+1}/{len(qa_pairs)}] Saved: {q[:60]}...")
                # Rate limit protection for the embedding API
                await asyncio.sleep(1.0)
            except Exception as e:
                print(f"[{i+1}/{len(qa_pairs)}] Error saving question: {e}")
                
    print(f"Successfully ingested {added} Q&A pairs from {os.path.basename(filepath)}!")

async def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Target all PDFs in the root directory
    pdf_files = [f for f in os.listdir(root_dir) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDFs found in {root_dir}")
        return
        
    for pdf in pdf_files:
        filepath = os.path.join(root_dir, pdf)
        await ingest_pdf(filepath)
        
    print("\n✅ All PDFs have been fully ingested into the vector database!")

if __name__ == "__main__":
    asyncio.run(main())
