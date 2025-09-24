from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

class Retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        query_embedding = self.embed(query)
        results = self.vector_store.query(query_embedding, top_k)
        return results

    def embed(self, text: str) -> np.ndarray:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode(text, convert_to_numpy=True)