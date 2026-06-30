from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
try:
    embeddings = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=["hello"],
        parameters={"input_type": "query", "truncate": "END"}
    )
    # The default for llama-text-embed-v2 is 1024, wait, let me try passing dimension 384. 
    # Actually wait, Pinecone's inference SDK accepts parameters. Let's see if dimension works.
    embeddings = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=["hello"],
        parameters={"dimension": 384, "input_type": "query"}
    )
    print("Embedding dimension:", len(embeddings.data[0].values))
except Exception as e:
    print("Error:", e)
