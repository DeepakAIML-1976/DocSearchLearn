# search.py
import os
import pickle
import numpy as np
from openai import OpenAI
import faiss
from config import OPENAI_API_KEY, FAISS_INDEX_FILE, DOC_MAP_FILE

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Paths
BASE_PATH = os.path.dirname(os.path.dirname(__file__))
EMBEDDINGS_FOLDER = os.path.join(BASE_PATH, "embeddings")
INDEX_PATH = os.path.join(EMBEDDINGS_FOLDER, FAISS_INDEX_FILE)
DOC_MAP_PATH = os.path.join(EMBEDDINGS_FOLDER, DOC_MAP_FILE)

# Load FAISS index and document map
index = faiss.read_index(INDEX_PATH)
with open(DOC_MAP_PATH, "rb") as f:
    doc_map = pickle.load(f)

# Function to get query embedding
def get_query_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)

def search(query, top_k=5):
    query_emb = get_query_embedding(query)
    distances, indices = index.search(query_emb, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        result = doc_map[idx]
        results.append({
            "rank": i + 1,
            "filename": result['filename'],
            "chunk": result['chunk'],
            "score": float(distances[0][i])
        })
    return results

if __name__ == "__main__":
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        results = search(query)
        print("\nTop Results:")
        for res in results:
            print(f"\nRank {res['rank']} | File: {res['filename']} | Score: {res['score']:.4f}")
            print(f"Snippet: {res['chunk'][:200]}...")
