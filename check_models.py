import os
import asyncio
from dotenv import load_dotenv
from groq import AsyncGroq
from google import genai

load_dotenv()

async def check_groq():
    print("--- GROQ MODELS ---")
    try:
        client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        models = await client.models.list()
        for m in models.data:
            print(m.id)
    except Exception as e:
        print("Groq error:", e)

def check_gemini():
    print("\n--- GEMINI MODELS ---")
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        for m in client.models.list():
            print(m.name)
    except Exception as e:
        print("Gemini error:", e)

async def main():
    await check_groq()
    check_gemini()

if __name__ == "__main__":
    asyncio.run(main())
