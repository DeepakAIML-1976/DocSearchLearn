# embedding.py
import numpy as np
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",  # or "text-embedding-3-large"
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)
