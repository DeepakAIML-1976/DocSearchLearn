# vector_store.py
import faiss
import numpy as np

def create_faiss_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

def save_index(index, path):
    faiss.write_index(index, path)

def load_index(path):
    return faiss.read_index(path)
