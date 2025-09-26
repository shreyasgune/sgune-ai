import os
import fitz
from docx import Document

def extract_text_from_pdf(path):
    try:
        text = ''
        with fitz.open(path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        print(f"Error processing PDF {path}: {e}")
        return None

def extract_text_from_docx(path):
    try:
        doc = Document(path)
        return '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error processing DOCX {path}: {e}")
        return None

def extract_text_from_txt(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            # Try different encodings if UTF-8 fails
            with open(path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            print(f"Error processing text file {path}: {e}")
            return None
    except Exception as e:
        print(f"Error processing text file {path}: {e}")
        return None

def is_text_file(filename):
    text_extensions = {'.txt', '.md', '.py', '.js', '.json', '.yaml', '.yml', '.ini', '.cfg', '.conf', '.log'}
    return any(filename.lower().endswith(ext) for ext in text_extensions)

def load_documents(folder):
    documents = []
    skipped_files = []
    processed_files = []
    
    print(f"\nScanning for documents in: {folder}")
    
    for root, _, files in os.walk(folder):
        for filename in files:
            path = os.path.join(root, filename)
            relative_path = os.path.relpath(path, folder)
            text = None
            
            try:
                if filename.lower().endswith('.pdf'):
                    print(f"Processing PDF: {relative_path}")
                    text = extract_text_from_pdf(path)
                elif filename.lower().endswith('.docx'):
                    print(f"Processing DOCX: {relative_path}")
                    text = extract_text_from_docx(path)
                elif is_text_file(filename):
                    print(f"Processing text file: {relative_path}")
                    text = extract_text_from_txt(path)
                
                if text:
                    metadata = {
                        'source': relative_path,
                        'type': os.path.splitext(filename)[1].lower()[1:],
                        'size': os.path.getsize(path),
                        'modified': os.path.getmtime(path)
                    }
                    documents.append({
                        'text': text,
                        'metadata': metadata
                    })
                    processed_files.append(relative_path)
                else:
                    skipped_files.append(relative_path)
            except Exception as e:
                print(f"Error processing {relative_path}: {e}")
                skipped_files.append(relative_path)
    
    print(f"\nProcessed {len(processed_files)} files:")
    for file in processed_files:
        print(f"✓ {file}")
    
    if skipped_files:
        print(f"\nSkipped {len(skipped_files)} files:")
        for file in skipped_files:
            print(f"⨯ {file}")
    
    return documents