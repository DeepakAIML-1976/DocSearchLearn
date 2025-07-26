import os
import pickle
import numpy as np
from tqdm import tqdm
from extract_text import extract_pdf_text, extract_docx_text, extract_excel_text
from chunking import chunk_text
from embedding import get_embedding
from vector_store import create_faiss_index, save_index
from config import FAISS_INDEX_FILE, DOC_MAP_FILE

# Base paths
BASE_PATH = os.path.dirname(os.path.dirname(__file__))
DATA_FOLDER = os.path.join(BASE_PATH, "data")
EMBEDDINGS_FOLDER = os.path.join(BASE_PATH, "embeddings")

# Ensure required folders exist
if not os.path.exists(DATA_FOLDER):
    raise FileNotFoundError(f"Data folder not found: {DATA_FOLDER}")

if not os.path.exists(EMBEDDINGS_FOLDER):
    os.makedirs(EMBEDDINGS_FOLDER)

# Collect supported files
supported_extensions = (".pdf", ".docx", ".xlsx")
files = [
    os.path.join(DATA_FOLDER, f)
    for f in os.listdir(DATA_FOLDER)
    if f.lower().endswith(supported_extensions)
]

if not files:
    raise FileNotFoundError(f"No supported files found in {DATA_FOLDER}")

print(f"Found {len(files)} documents. Starting ingestion...\n")

embeddings = []
doc_map = []

for file_path in tqdm(files, desc="Processing files"):
    try:
        # Extract text based on file type
        text = ""
        if file_path.endswith(".pdf"):
            text = extract_pdf_text(file_path)
        elif file_path.endswith(".docx"):
            text = extract_docx_text(file_path)
        elif file_path.endswith(".xlsx"):
            text = extract_excel_text(file_path)
        else:
            print(f"⚠ Unsupported file type: {file_path}")
            continue

        if not text.strip():
            print(f"⚠ No text extracted from: {file_path}")
            continue

        # Split into chunks and generate embeddings
        chunks = chunk_text(text)
        for chunk in chunks:
            emb = get_embedding(chunk)
            embeddings.append(emb)
            doc_map.append({
                "chunk": chunk,
                "filename": os.path.basename(file_path)
            })

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        continue

# Check if we have any embeddings
if len(embeddings) == 0:
    print("\n⚠ No embeddings generated. Check your documents or API key.")
    exit()

# Convert embeddings to NumPy array
embeddings_array = np.array(embeddings).astype("float32")

# Create and save FAISS index
index = create_faiss_index(embeddings_array)
save_index(index, os.path.join(EMBEDDINGS_FOLDER, FAISS_INDEX_FILE))

# Save document map
with open(os.path.join(EMBEDDINGS_FOLDER, DOC_MAP_FILE), "wb") as f:
    pickle.dump(doc_map, f)

print(f"\n✅ Ingestion complete!")
print(f"FAISS index saved at {os.path.join(EMBEDDINGS_FOLDER, FAISS_INDEX_FILE)}")
print(f"Document map saved at {os.path.join(EMBEDDINGS_FOLDER, DOC_MAP_FILE)}")
