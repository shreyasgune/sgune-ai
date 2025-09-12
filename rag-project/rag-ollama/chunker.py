from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_texts(texts, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks
