from document_loader import load_documents
from chunker import chunk_texts
from embed_and_index import build_faiss_index
from retriever import search_index
from ollama_qa import generate_answer_ollama

def main():
    docs = load_documents('./docs')
    chunks = chunk_texts(docs)

    index, chunk_list = build_faiss_index(chunks)

    query = input("Ask a question about your docs: ")
    top_chunks = search_index(query, index, chunk_list)

    context = "\n\n".join(top_chunks)
    answer = generate_answer_ollama(query, context)

    print("Answer::")
    print(answer)

if __name__ == "__main__":
    main()