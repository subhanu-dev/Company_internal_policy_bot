from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
import faiss

load_dotenv()
openai_key = os.getenv("OPENAI_KEY")

client = OpenAI(api_key=openai_key)

folder_path = "c:\web-dev\RAG Chatbot\documents"
file_list = os.listdir(folder_path)

text_files = []


def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


all_chunks = list()
for filename in file_list:
    if filename.endswith(".txt") or filename.endswith(".md"):
        text_files.append(filename)
        with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
            text = f.read()
            chunks = chunk_text(text)
            all_chunks.extend(chunks)


# checking text files and chunking a test document.

# print(text_files)

# with open("C:\web-dev\RAG Chatbot\documents\conduct.txt", "r", encoding="utf-8") as f:
#     text = f.read()
#     chunks = chunk_text(text)

# print(len(chunks))

# for i, each in enumerate(chunks):
#     print(f" {i} th chunk is \n {each}")


# print(all_chunks)
print("count of all chunks:", len(all_chunks))

embeddings = []


def generate_embeddings():
    for chunk in all_chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",  # OpenAI's small v3 embedding model
            input=chunk,
        )
        embeddings.append(response.data[0].embedding)


# Print embedding info
print(len(embeddings))

# for i, emb in enumerate(embeddings):
#     print(f"Chunk {i} embedding length: {len(emb)}")

# print(embeddings[0])
