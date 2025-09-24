from pydantic import BaseModel
from typing import List, Optional

class Document(BaseModel):
    id: str
    content: str
    embeddings: List[float]

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    documents: List[Document]