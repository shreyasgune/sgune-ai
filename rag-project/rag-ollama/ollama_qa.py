import ollama

def generate_answer_ollama(query, context, model="mistral"):
    prompt = f"""
You are a helpful assistant. User the context below to answer the question.

Context:
{context}

Question: {query}

Answer:
"""
    response = ollama.chat(model=model, messages=[
        {"role": "user", "content": prompt}
    ])
    return response['message']['content']