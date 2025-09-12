import os
import fitz
from docx import Document

def extract_text_from_pdf(path):
    text = ''
    with fitz.open(path) as doc:
        for page in  doc:
            text += page.get_text()
    return text

def extract_text_form_docx(path):
    doc = Document(path)
    return '\n'.join([para.text for para in doc.paragraphs])

def extract_text_from_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
    
def load_documents(folder):
    texts = []
    for filename in os.listdir(folder):
        path = os.path.join(folder,filename)
        if filename.endswith('.pdf'):
            texts.append(extract_text_from_pdf(path))
        elif filename.endswith('.docx'):
            texts.append(extract_text_form_docx(path))
        elif  filename.endswith('.txt'):
            texts.append(extract_text_from_txt(path))
    return texts