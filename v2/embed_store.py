from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
import faiss
import pickle


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


generate_embeddings()

# Print embedding info
print(len(embeddings))

# for i, emb in enumerate(embeddings):
#     print(f"Chunk {i} embedding length: {len(emb)}")

# print(embeddings[0])


embeddings = np.array(embeddings, dtype=np.float32)


def l2_normalize(vec):
    return vec / np.linalg.norm(vec)


normalized_embeddings = np.array(
    [l2_normalize(e) for e in embeddings], dtype=np.float32
)

print(normalized_embeddings[0])  # taking a look at normalized embeddings

# writing embeddings into vector dbs.

embedding_dim = normalized_embeddings.shape[1]
# print(embedding_dim) # 1536i bn

index = faiss.IndexFlatIP(
    embedding_dim
)  # Inner product = cosine similarity for normalized vectors
index.add(normalized_embeddings)

index_path = "C:\\web-dev\\RAG Chatbot\\v2\\faiss\\vector_index.faiss"

faiss.write_index(index, index_path)

print(f"FAISS index saved to {index_path}")

# Save the chunks that correspond to the vectors by position
chunks_path = "C:\\web-dev\\RAG Chatbot\\v2\\faiss\\chunks.pkl"
with open(chunks_path, "wb") as f:
    pickle.dump(all_chunks, f)
print(f"Chunks saved to {chunks_path}")

"""
chunks.pkl file that stores your text chunks in the same order as they 
appear in the FAISS index.
 When you search the index later, the indices returned will match the positions in this chunks file.
"""
