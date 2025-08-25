import faiss
from openai import OpenAI
import pickle
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_KEY")
client = OpenAI(api_key=openai_key)


# Paths to saved files
index_path = "C:\\web-dev\\RAG Chatbot\\v2\\faiss\\vector_index.faiss"
chunks_path = "C:\\web-dev\\RAG Chatbot\\v2\\faiss\\chunks.pkl"


index = faiss.read_index(index_path)

# Load the text chunks
with open(chunks_path, "rb") as f:
    chunks = pickle.load(f)


def l2_normalize(vec):
    """Normalize vector to unit length"""
    return vec / np.linalg.norm(vec)


def search_similar_chunks(query, k=3, threshold=0.20):
    """
    Search for chunks similar to the query

    Args:
        query (str): The query text
        k (int): Number of results to return

    Returns:
        list: Top k similar chunks with their scores
    """
    # Generate embedding for the query
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    )
    query_embedding = np.array([response.data[0].embedding], dtype=np.float32)

    # Normalize the query embedding
    normalized_query = np.array([l2_normalize(query_embedding[0])], dtype=np.float32)

    # Search the index
    distances, indices = index.search(normalized_query, k)
    # print(indices)
    results = []

    for i, idx in enumerate(indices[0]):
        # print(i, idx)
        if idx != -1:  # Valid index
            score = float(distances[0][i])
            if score >= threshold:
                results.append(
                    {
                        "chunk": chunks[idx],
                        "score": float(distances[0][i]),
                        "index": int(idx),
                    }
                )

    return results


print(search_similar_chunks("teach me how to cook"))
