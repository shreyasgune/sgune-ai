from typing import List, Any
from your_vector_database_library import VectorDatabase  # Replace with actual library import
from src.core.embed_and_index import embed

class VectorStore:
    def __init__(self, db_url: str):
        self.db = VectorDatabase(db_url)

    def add_embeddings(self, texts: List[str]) -> None:
        embeddings = [embed(text) for text in texts]
        for embedding in embeddings:
            self.db.add(embedding)

    def query(self, query_text: str, top_k: int = 5) -> List[Any]:
        query_embedding = embed(query_text)
        return self.db.query(query_embedding, top_k)