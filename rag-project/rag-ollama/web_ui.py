import gradio as gr

from document_loader import load_documents
from chunker import chunk_texts
from embed_and_index import build_faiss_index
from retriever import search_index
from ollama_qa import generate_answer_ollama

print("Loading documents")
docs = load_documents('./docs')
chunks = chunk_texts(docs)
index, chunk_list = build_faiss_index(chunks)
print("Index built.")

def ask_question(query):
    if not query.strip():
        return "Please enter a question"
    relevant_chunks = search_index(query, index, chunk_list)
    context = "\n\n".join(relevant_chunks)
    answer = generate_answer_ollama(query, context)
    return answer

iface = gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(lines=2, placeholder="Ask away", label="Your Question"),
    outputs=gr.Textbox(label="Answer"),
    title="RAG for sgune",
    description="Ask Away"
)

if __name__ == "__main__":
    iface.launch()
