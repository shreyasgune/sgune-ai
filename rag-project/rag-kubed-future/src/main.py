from fastapi import FastAPI
from src.api.routes import router as api_router
from src.core.config import settings

app = FastAPI(title="RAG Project API", version="1.0")

# Include the API routes
app.include_router(api_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG Project API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)