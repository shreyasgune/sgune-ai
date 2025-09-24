from document_loader import load_documents
from chunker import chunk_texts
from embed_and_index import build_faiss_index
from retriever import search_index
from ollama_qa import generate_answer_ollama
from vectordb import VectorDB, Document

def inMemory():
    docs = load_documents('./docs')
    chunks = chunk_texts(docs)

    index, chunk_list = build_faiss_index(chunks)

    query = input("Ask a question about your docs: ")
    top_chunks = search_index(query, index, chunk_list)

    context = "\n\n".join(top_chunks)
    answer = generate_answer_ollama(query, context)

    print("Answer::")
    print(answer)



def vect():
    # Using Vector DB
    db = VectorDB()

    try:
        db.load()
        print("Loaded existing vector Database")
    except:
        print("Creating new vector database")
        texts = load_documents('./docs')
        chunks = chunk_texts(texts)

        documents = [Document(text=chunk) for chunk in chunks]
        db.add_documents(documentso)
        db.save()
    
    while True:
        query = input("Ask a question (or 'quit' to exit):")
        if query.lower() == 'quit':
            break

        results = db.search(query, k=3)
        context = "\n\n".join([doc.text for doc, score in results])

        answer = generate_answer_ollama(query, context)
        print("\nAnswer:", answer)
        print("\nSources:")
        for doc, score in results:
            print(f"- Score {score: .4f}")
            print(f"   {doc.text[:200]}...\n")

def main():
    # inMemory()
    vect()

if __name__ == "__main__":
    main()