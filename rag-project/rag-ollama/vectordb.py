import os
import json
import faiss #A library for fast similarity search—used to quickly find similar things (like texts).
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass #Lets us define simple data structures (like Document).
from sentence_transformers import SentenceTransformer #Imports a model that turns text into numbers (vectors) that capture meaning.

@dataclass #This automatically creates helpful things like __init__() so we don’t have to write boring boilerplate code.
# so that's a decorator and it avoids us having to write more code
# __init__() → the constructor
# __repr__() → nice string representation
# __eq__() → comparison logic

class Document:
    text: str #actual content of the text
    metadata: Dict = None #extra info like tags, source etc

# Our goal is to turn docs into vectors, store them and search through them.
class VectorDB:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', index_path: str = 'vector_store'):
        # we are using "self" cuz we need to point to the same object when calling the instance from the class
        self.model = SentenceTransformer(model_name)
        self.index_path = Path(index_path)
        self.index_path.mkdir(exist_ok=True)
        self.index = None #will hold the serachable db
        self.documents = [] #this will hold document objects, its ordered so index of the doc is the sae as the index of its vector in FAISS index, can also append and pop stuff
        self.dimension = None # number of values in each vector

    def embed(self, text:str) -> np.ndarray:  #the arrow thing here is basically saying that the function will return something of type Array
        return self.model.encode(text, convert_to_numpy=True)
    
    def add_documents(self,documents: List[Document]) -> None:
        embeddings = []
        for doc in documents:
            embedding = self.embed(doc.text) #stores doclist in engine
            if self.dimension is None:
                self.dimension = len(embedding)
                self.index = faiss.IndexFlatL2(self.dimension)
            embeddings.append(embedding)
            self.documents.append(doc)
        
        if embeddings:
            self.index.add(np.array(embeddings).astype('float32'))
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]: #that tuple is basically returning (Document object, distance score)
        #why tuple? cuz its ordered and of fixed size, cuz we don't need to modify the content, so its faster and safer.
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
            try:
                faiss.write_index(self.index, str(self.index_path / 'index.faiss'))

                documents_data = [
                    { 
                        'text': doc.text,
                        'metadata': doc.metadata
                    }
                    for doc in self.documents
                ]
                with open(self.index_path / 'documents.json', 'w') as f:
                    json.dump(documents_data, f)
            except Exception as e:
                print(f"Error saving vector database: {e}")
                raise

    def load(self) -> None:
        index_file = self.index_path / 'index.faiss'
        documents_file = self.index_path / 'documents.json'

        if not index_file.exists() or not documents_file.exists():
            raise FileNotFoundError(f"Vector database files not found at {self.index_path}")

        try:
            self.index = faiss.read_index(str(index_file))
            self.dimension = self.index.d

            with open(documents_file) as f:
                documents_data = json.load(f)

            self.documents = [
                Document(
                    text=doc['text'],
                    metadata=doc['metadata']
                )
                for doc in documents_data                
            ]
        except Exception as e:
            print(f"Error loading vector database: {e}")
            raise
