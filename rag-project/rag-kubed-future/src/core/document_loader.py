import os

def load_document(file_path):
    """Load a document from the specified file path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    return content

def load_documents_from_directory(directory_path):
    """Load all documents from the specified directory."""
    documents = {}
    
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"The path {directory_path} is not a directory.")
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            documents[filename] = load_document(file_path)
    
    return documents