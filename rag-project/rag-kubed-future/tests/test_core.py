import pytest
from src.core.chunker import chunk_text
from src.core.document_loader import load_document
from src.core.embed_and_index import embed, build_faiss_index

def test_chunk_text():
    text = "This is a test document that needs to be chunked."
    chunks = chunk_text(text, chunk_size=10)
    assert len(chunks) == 5  # Adjust based on expected chunking logic
    assert all(len(chunk) <= 10 for chunk in chunks)

def test_load_document():
    document = load_document("path/to/test/document.txt")
    assert document is not None
    assert isinstance(document, str)  # Assuming the document is loaded as a string

def test_embed():
    text = "Hello, world!"
    embedding = embed(text)
    assert embedding is not None
    assert len(embedding) > 0  # Ensure that the embedding is generated

def test_build_faiss_index():
    chunks = ["chunk1", "chunk2", "chunk3"]
    index, _ = build_faiss_index(chunks)
    assert index is not None
    assert index.ntotal == len(chunks)  # Ensure all chunks are added to the index