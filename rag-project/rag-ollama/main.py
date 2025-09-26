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
        print("Loaded existing vector database")
    except FileNotFoundError:
        print("\nCreating new vector database...")
        try:
            # Load all documents with metadata
            loaded_docs = load_documents('./docs')
            print(f"\nFound {len(loaded_docs)} documents")
            
            # Process chunks
            print("\nChunking documents...")
            all_chunks = []
            for doc in loaded_docs:
                chunks = chunk_texts([doc['text']])  # Chunk each document individually
                for chunk in chunks:
                    all_chunks.append(Document(
                        text=chunk,
                        metadata=doc['metadata']  # Preserve the source document's metadata
                    ))
            print(f"Created {len(all_chunks)} chunks")

            # Add to vector database
            print("\nAdding documents to vector database...")
            db.add_documents(all_chunks)
            
            # Save the database
            print("\nSaving vector database...")
            db.save()
            print("Vector database created and saved successfully")
            
        except Exception as e:
            print(f"Error creating vector database: {e}")
            return
    except Exception as e:
        print(f"Error loading vector database: {e}")
        return
    
    while True:
        query = input("Ask a question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break

        try:
            results = db.search(query, k=3)
            if not results:
                print("\nNo relevant results found.")
                continue

            context = "\n\n".join([doc.text for doc, score in results])
            print("\nGenerating answer...")
            
            answer = generate_answer_ollama(query, context)
            print("\nAnswer:", answer)
            print("\nSources:")
            for doc, score in results:
                source = doc.metadata.get('source', 'Unknown source') if doc.metadata else 'Unknown source'
                print(f"\n- Source: {source}")
                print(f"  Relevance score: {score:.4f}")
                print(f"  Preview: {doc.text[:200]}...")
        except Exception as e:
            print(f"Error processing query: {e}")

def main():
    # inMemory()
    vect()

if __name__ == "__main__":
    main()