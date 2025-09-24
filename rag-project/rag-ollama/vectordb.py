import os
import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

@dataclass
class Document:
    text: str
    metadata: Dict = None

class VectorDB:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', index_path: str = 'vector_store'):
        self.model = SentenceTransformer(model_name)
        self.index_path = Path(index_path)
        self.index_path.mkdir(exist_ok=True)
        self.index = None
        self.documents = []
        self.dimension = None

    def embed(self, text:str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)
    
    def add_documents(self,documents: List[Document]) -> None:
        embeddings = []
        for doc in documents:
            embedding = self.embed(doc.text)
            if self.dimension is None:
                self.dimension = len(embedding)
                self.index = faiss.IndexFlatL2(self.dimension)
            embeddings.append(embedding)
            self.documents.append(doc)
        
        if embeddings:
            self.index.add(np.array(embeddings).astype('float32'))
    
    def search(self, query: str, k: int = 5) -> List[Tuple(Document, float)]:
        query_vector = self.embed(query)
        query_vector = np.array([query_vector]).astype('float32')
        distances, indices = self.index.search(query_vector, k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1:
                results.append((self.documents[idx], float(distance)))
        return results
    
    def save(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path / 'index.faiss'))

            documents_data = [
                { 
                    'text': doc.text,
                    'metadata': doc.metadata
                }
                for doc in self.documents
            ]
            with open(self.index_path / 'douments.json', 'w') as f:
                json.dump(documents_data, f)

    def load(self) -> None:
        index_file = self.index_path / 'index.faiss'
        documents_file = self.index_path / 'documents.json'

        if index_file.exists() and documents_file.exists():
            self.index = faiss.read_index(str(index_file))
            self.dimension = self.index.d

            with open(documents_file) as f:
                documents_data = json.load(f)

            self.documents =  [
                Document(
                    text=doc['text'],
                    metadata=doc['metadata']
                )
                for doc in documents_data                
            ]
