import os
import json
import asyncio
from groq import AsyncGroq
from dotenv import load_dotenv
from ddgs import DDGS
from pinecone import Pinecone

load_dotenv()

# Initialize Groq client
# Check to avoid crashing if key is missing on startup
groq_api_key = os.getenv("GROQ_API_KEY", "")
groq_client = AsyncGroq(api_key=groq_api_key) if groq_api_key else None

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("interview-qa")

async def get_embedding(text: str):
    # Use Pinecone's blazing fast Inference API instead of HuggingFace!
    try:
        response = await asyncio.to_thread(
            pc.inference.embed,
            model="llama-text-embed-v2",
            inputs=[text],
            parameters={"dimension": 384, "input_type": "query"}
        )
        return response.data[0].values
    except Exception as e:
        print(f"Pinecone embedding failed: {e}")
        raise Exception("Failed to generate embedding via Pinecone.")

async def combine_answers(question: str, old_answer: str, new_answer: str) -> str:
    """
    Uses Groq LLM to summarize and combine the existing and new answers.
    """
    if not groq_client:
        # Fallback if no API key is provided
        return f"{old_answer}\n\n---\n\n{new_answer}"

    prompt = f"""
You are an expert technical interviewer and educator. 
A student has asked a question: "{question}"

I have two potential answers for this question. 
Answer 1: {old_answer}
Answer 2: {new_answer}

Please combine, refine, and summarize these two answers into a single, comprehensive, and well-structured answer. Ensure all key points are covered accurately.
"""
    try:
        chat_completion = await groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="qwen/qwen3-32b",
            temperature=0.3,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return f"{old_answer}\n\n---\n\n{new_answer}"


async def web_search(query: str) -> str:
    """Helper function to perform web search using DuckDuckGo."""
    try:
        def sync_search():
            return DDGS().text(query, max_results=3)
        results = await asyncio.to_thread(sync_search)
        if not results:
            return "No results found."
        return "\n\n".join([f"Title: {r['title']}\nSnippet: {r['body']}\nSource: {r['href']}" for r in results])
    except Exception as e:
        return f"Web search failed: {e}"

async def generate_rag_answer(user_query: str, retrieved_docs: list) -> str:
    """
    Uses Groq LLM to answer a user's query based on retrieved contexts.
    If the context lacks the answer, uses the web_search tool to find it.
    """
    if not groq_client:
        return "GROQ_API_KEY not configured. Cannot generate a response."

    context = "\n\n".join([f"Q: {doc['questions']}\nA: {doc['answer']}" for doc in retrieved_docs])
    
    prompt = f"""You are an intelligent interview preparation assistant. 
You have been provided with the following Context containing the user's saved interview questions and answers.

CRITICAL RULES:
1. You MUST prioritize the Context. If the Context contains ANY information relevant to the user's query, use it to answer the question and DO NOT use the web_search tool.
2. ONLY use the `web_search` tool if the Context is completely empty or completely irrelevant to the user's query.
3. Do not use the web_search tool just to "expand" or "supplement" information if the Context already has a relevant answer.
4. DO NOT hallucinate. 

Context:
{context}
"""
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Perform a web search to find current information or answers not available in the local context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to run."
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_query}
    ]

    try:
        # Allow up to 3 tool call iterations to support complex queries
        for iteration in range(3):
            response = await groq_client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.3,
            )
            
            response_message = response.choices[0].message
            
            # Check if the LLM decided to use the tool
            if response_message.tool_calls:
                # We must append the LLM's response first as a properly formatted dictionary
                messages.append({
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": t.id,
                            "type": "function",
                            "function": {
                                "name": t.function.name,
                                "arguments": t.function.arguments
                            }
                        } for t in response_message.tool_calls
                    ]
                })
                
                for tool_call in response_message.tool_calls:
                    if tool_call.function.name == "web_search":
                        args = json.loads(tool_call.function.arguments)
                        print(f"Executing web search for: {args['query']}")
                        search_result = await web_search(args["query"])
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": "web_search",
                            "content": search_result
                        })
                
                # Force the LLM to stop searching if it has already searched
                messages.append({
                    "role": "system",
                    "content": "You have received the web search results. You MUST NOT use the web_search tool again. Provide your final answer to the user immediately."
                })
                
                # Continue loop to let LLM process the tool result
                continue
                
            # If no tool was called (or we finished calling tools), return the direct response
            return response_message.content
            
        return "I performed several searches, but the results required too much complex searching to compile a final answer."
        
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return "An error occurred while generating the response."


async def generate_answer_for_question(question: str) -> str:
    """
    Uses Groq LLM to generate a comprehensive answer for a single interview question.
    """
    if not groq_client:
        return "GROQ_API_KEY not configured. Cannot generate an answer."

    prompt = f"""
You are an expert technical interviewer and educator. 
A student has asked the following interview question: "{question}"

Please provide a clear, accurate, and comprehensive answer to this question. 
Format your response nicely, using bullet points or paragraphs as appropriate.
Do not include any conversational filler, just the answer.
"""
    try:
        chat_completion = await groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful expert answering technical questions."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="qwen/qwen3-32b",
            temperature=0.3,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return "An error occurred while generating the answer."

async def parse_pdf_text_to_qa(text: str) -> list:
    """
    Uses Groq LLM to extract Q&A pairs from raw text.
    Processes text in chunks to handle multi-page PDFs.
    """
    if not groq_client:
        print("No Groq API key")
        return []

    all_qa_pairs = []
    chunk_size = 15000
    
    # Split text into chunks
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    for chunk in chunks:
        prompt = f"""
You are an intelligent document parsing assistant for an Interview Preparation platform. 
I have extracted the following text from a PDF document.

Extract each question and its corresponding answer and return them as JSON.
CRITICAL INSTRUCTION: You MUST ONLY extract questions and answers that are related to interviews, technical concepts, or professional skills. Ignore general knowledge, irrelevant content, or conversational filler.
The JSON object MUST have a single key "qa_pairs" which is an array of objects.
Each object in the array MUST have two keys: "question" and "answer".
If you cannot find any relevant interview questions or answers, return {{"qa_pairs": []}}.

Text:
{chunk}
"""
        try:
            chat_completion = await groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a JSON parsing assistant. You always output a valid JSON object."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="qwen/qwen3-32b",
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            response_text = chat_completion.choices[0].message.content.strip()
            parsed = json.loads(response_text)
            chunk_pairs = parsed.get("qa_pairs", [])
            all_qa_pairs.extend(chunk_pairs)
                
        except Exception as e:
            print(f"Error calling Groq API for chunk: {e}")
            continue

    return all_qa_pairs

async def is_interview_related(text: str) -> bool:
    """
    Uses Groq LLM to quickly classify if the input text is related to interviews or professional skills.
    """
    if not groq_client:
        return True # Fallback if no API key

    prompt = f"""
You are an AI classification system. You must determine if the following text is related to interview preparation, job interviews, technical concepts, or professional career skills.
Respond with ONLY "YES" if it is related, or "NO" if it is general knowledge, inappropriate, or irrelevant.

Text: "{text[:1000]}"
"""
    try:
        chat_completion = await groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="qwen/qwen3-32b",
            temperature=0.1,
            max_tokens=10
        )
        response = chat_completion.choices[0].message.content.strip().upper()
        return "YES" in response
    except Exception as e:
        print(f"Error calling Groq API for classification: {e}")
        return True

async def parse_image_to_qa(base64_image: str, mime_type: str) -> list:
    """
    Uses Groq's multimodal LLM to extract Q&A pairs from an image.
    """
    if not groq_client:
        print("No Groq API key")
        return []

    prompt = """
You are an intelligent document parsing assistant for an Interview Preparation platform. 
I have uploaded an image (e.g., a screenshot, whiteboard, or slide).

Extract each question and its corresponding answer and return them as JSON.
CRITICAL INSTRUCTION: You MUST ONLY extract questions and answers that are related to interviews, technical concepts, or professional skills. Ignore general knowledge or irrelevant content.
The JSON object MUST have a single key "qa_pairs" which is an array of objects.
Each object in the array MUST have two keys: "question" and "answer".
If you cannot find any relevant interview questions or answers, return {"qa_pairs": []}.
"""
    try:
        chat_completion = await groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        response_text = chat_completion.choices[0].message.content.strip()
        parsed = json.loads(response_text)
        return parsed.get("qa_pairs", [])
            
    except Exception as e:
        print(f"Error calling Groq Vision API: {e}")
        return []
