from typing import List

def chunk_text(text: str, chunk_size: int) -> List[str]:
    """Splits the input text into chunks of specified size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def chunk_documents(documents: List[str], chunk_size: int) -> List[List[str]]:
    """Chunks a list of documents into smaller pieces."""
    return [chunk_text(doc, chunk_size) for doc in documents]