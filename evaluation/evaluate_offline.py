import os
import sys
import json
import time
import asyncio
import argparse
from typing import List, Dict, Any

# Add the backend directory to sys.path so we can import rag and database
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import rag
import database
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from google import genai
from google.genai import types

# Set up Gemini Judge
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=GEMINI_API_KEY)

def evaluate_with_gemini(question: str, generated_answer: str, ground_truth: str, context: str) -> dict:
    """
    Uses Gemini (via OpenRouter) to score Answer Correctness and Faithfulness (0-5 scale).
    """
    prompt = f"""
    You are an expert AI evaluator grading a Retrieval-Augmented Generation (RAG) system.
    
    You will be provided with:
    1. The user's Question
    2. The Retrieved Context (the chunks found in the database)
    3. The Ground Truth Answer (the gold standard answer)
    4. The Generated Answer (produced by the RAG system)
    
    Please evaluate the Generated Answer on two metrics:
    
    A. Faithfulness (Score 0-5): 
       Does the Generated Answer stick strictly to the facts presented in the Retrieved Context? 
       (5 = Completely faithful, 0 = High hallucination or contradicts context)
       Note: If the Retrieved Context is empty but the model says "I don't know" or performs a web search properly as instructed by its prompt, give a high faithfulness score.
       
    B. Answer Correctness (Score 0-5):
       How semantically accurate is the Generated Answer compared to the Ground Truth Answer?
       (5 = Perfect match in meaning, 0 = Completely incorrect)
       
    Respond with ONLY a JSON object in this exact format, with no markdown formatting or extra text:
    {{
        "faithfulness": <int>,
        "correctness": <int>,
        "reasoning": "<short 1-sentence reasoning>"
    }}
    
    Question: {question}
    Retrieved Context: {context}
    Ground Truth Answer: {ground_truth}
    Generated Answer: {generated_answer}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0
            )
        )
        text = response.text.strip()
        # Strip markdown if present
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text.strip())
    except Exception as e:
        print(f"Error calling Gemini via OpenRouter: {e}")
        return {"faithfulness": 0, "correctness": 0, "reasoning": str(e)}

async def run_evaluation(limit: int = None):
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'golden_dataset_full.json'))
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return
        
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    if limit:
        dataset = dataset[:limit]
        
    print(f"Starting evaluation on {len(dataset)} questions...")
    print("Note: Enforcing rate limit of 15 requests/minute for Gemini (4.1 seconds delay per request).")
    
    total_hit = 0
    total_faithfulness = 0
    total_correctness = 0
    
    for i, item in enumerate(dataset):
        print(f"--- Evaluating {i+1}/{len(dataset)} ---")
        question = item['question']
        ground_truth = item['ground_truth_answer']
        context_snippet = item.get('context_snippet', '')
        
        print(f"Q: {question}")
        
        # 1. Retrieval
        try:
            embedding = await rag.get_embedding(question)
            results = await asyncio.to_thread(rag.pinecone_index.query, vector=embedding, top_k=3, include_values=False)
            
            retrieved_docs = []
            hit = False
            
            if results and results.matches:
                for match in results.matches:
                    resp = await asyncio.to_thread(database.supabase.table("qa_records").select("*").eq("id", int(match.id)).execute)
                    if resp.data:
                        r = resp.data[0]
                        qs = json.loads(r["questions_json"]) if isinstance(r["questions_json"], str) else r["questions_json"]
                        retrieved_docs.append({
                            "questions": qs,
                            "answer": r["answer"]
                        })
                        
                        # Check Hit: if the expected context snippet is roughly in the retrieved answer or question
                        if context_snippet and len(context_snippet) > 10:
                            # snippet might be truncated, check first 50 chars
                            check_str = context_snippet[:50].lower()
                            if check_str in str(qs).lower() or check_str in r["answer"].lower():
                                hit = True
            
            if hit:
                total_hit += 1
                print("Retrieval: HIT ✅")
            else:
                print("Retrieval: MISS ❌")
                
            # 2. Generation
            generated_answer = await rag.generate_rag_answer(question, retrieved_docs)
            
            context_str = "\n\n".join([f"Q: {doc['questions']}\nA: {doc['answer']}" for doc in retrieved_docs])
            
            # 3. Evaluation (Judge)
            eval_result = evaluate_with_gemini(question, generated_answer, ground_truth, context_str)
            
            faith_score = eval_result.get("faithfulness", 0)
            corr_score = eval_result.get("correctness", 0)
            reasoning = eval_result.get("reasoning", "No reasoning provided")
            
            total_faithfulness += faith_score
            total_correctness += corr_score
            
            print(f"Faithfulness: {faith_score}/5 | Correctness: {corr_score}/5")
            print(f"Reasoning: {reasoning}")
            
        except Exception as e:
            print(f"Error during evaluation of question {i+1}: {e}")
            
        print("-" * 40)
        
        # Strict rate limit: 10 requests per minute = 1 request every 6 seconds
        # Sleep for 6 seconds before the next iteration
        # Rate limiting (Gemini free tier is ~15 RPM max)
        time.sleep(4.1)
            
    # Final Report
    print("\n" + "="*40)
    print("🎉 EVALUATION COMPLETE 🎉")
    print("="*40)
    print(f"Total Questions Evaluated: {len(dataset)}")
    print(f"Hit Rate (Retrieval): {(total_hit / len(dataset)) * 100:.2f}%")
    print(f"Average Correctness: {(total_correctness / len(dataset)):.2f} / 5.0")
    print(f"Average Faithfulness: {(total_faithfulness / len(dataset)):.2f} / 5.0")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Offline Evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of test samples")
    args = parser.parse_args()
    
    asyncio.run(run_evaluation(limit=args.limit))
