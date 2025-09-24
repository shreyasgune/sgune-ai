from fastapi import APIRouter
from src.api.models import YourModel  # Replace with your actual model
from src.core.embed_and_index import embed, build_faiss_index  # Import your functions
from src.services.vector_store import add_embedding, query_embedding  # Import your vector store functions

router = APIRouter()

@router.post("/embed")
async def create_embedding(data: YourModel):
    embedding = embed(data.text)
    # Save the embedding to the vector store
    add_embedding(embedding)
    return {"embedding": embedding}

@router.post("/query")
async def query_embeddings(data: YourModel):
    results = query_embedding(data.text)
    return {"results": results}

# Add more routes as needed