from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import pickle
import numpy as np
import faiss
from openai import OpenAI
import csv
from datetime import datetime
from src.config import OPENAI_API_KEY, FAISS_INDEX_FILE, DOC_MAP_FILE

app = FastAPI(title="Oil & Gas AI Search API")

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=OPENAI_API_KEY)

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
EMBEDDINGS_FOLDER = os.path.join(BASE_PATH, "embeddings")
DATA_FOLDER = os.path.join(BASE_PATH, "data")
LOG_FOLDER = os.path.join(BASE_PATH, "logs")

if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

FEEDBACK_FILE = os.path.join(LOG_FOLDER, "feedback.csv")
DOWNLOAD_FILE = os.path.join(LOG_FOLDER, "downloads.csv")

INDEX_PATH = os.path.join(EMBEDDINGS_FOLDER, FAISS_INDEX_FILE)
DOC_MAP_PATH = os.path.join(EMBEDDINGS_FOLDER, DOC_MAP_FILE)

index = faiss.read_index(INDEX_PATH)
with open(DOC_MAP_PATH, "rb") as f:
    doc_map = pickle.load(f)

feedback_scores = {}

def get_query_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)

def generate_summary(context, query):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in Oil & Gas plant design. Provide precise answers."},
            {"role": "user", "content": f"Previous Context:\n{context}\nNew Query: {query}"}
        ],
        temperature=0.3,
        max_tokens=300
    )
    return response.choices[0].message.content

def update_feedback_cache(filename, feedback):
    if filename not in feedback_scores:
        feedback_scores[filename] = 0
    feedback_scores[filename] += 1 if feedback == "positive" else -1

@app.get("/search")
async def search(query: str = Query(...), top_k: int = 5, context: str = ""):
    query_emb = get_query_embedding(query)
    distances, indices = index.search(query_emb, top_k)
    results = []
    combined_text = ""

    for i, idx in enumerate(indices[0]):
        res = doc_map[idx]
        filename = res['filename']
        score = float(distances[0][i])
        score -= 0.1 * feedback_scores.get(filename, 0)  # Apply feedback boost
        results.append({
            "rank": i + 1,
            "filename": filename,
            "snippet": res['chunk'][:300],
            "score": score
        })
        combined_text += res['chunk'] + "\n"

    results = sorted(results, key=lambda x: x['score'])
    summary = generate_summary(context + "\n" + combined_text, query)

    return {"query": query, "summary": summary, "results": results}

@app.get("/download")
async def download(filename: str):
    file_path = os.path.join(DATA_FOLDER, filename)
    if os.path.exists(file_path):
        with open(DOWNLOAD_FILE, "a", newline="") as f:
            writer = csv.writer
