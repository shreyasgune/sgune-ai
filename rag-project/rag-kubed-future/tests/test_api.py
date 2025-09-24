import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the API!"}

def test_embed_endpoint():
    response = client.post("/embed", json={"text": "Hello, world!"})
    assert response.status_code == 200
    assert "embedding" in response.json()

def test_query_endpoint():
    response = client.post("/query", json={"query": "What is AI?"})
    assert response.status_code == 200
    assert "results" in response.json()