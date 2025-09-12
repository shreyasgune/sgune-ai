import numpy as np
from embed_and_index import embed

def search_index(query, index, chunks, k=5):
    query_embedding = np.array([embed(query)]).astype('float32')
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]