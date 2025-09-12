import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
model = model.to('cuda')

def embed(text):
    return model.encode(text, convert_to_numpy=True)

def build_faiss_index(chunks):
    embeddings = [embed(chunk) for chunk in chunks]
    dimension = len(embeddings[0])
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(dimension))
    # index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index, chunks

