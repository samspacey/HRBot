import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_pdfs_from_folder(folder_path: str):
    # Get all PDF files in the folder
    pdf_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

    # Setup chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_chunks = []

    for path in pdf_paths:
        print(f"Loading: {path}")
        loader = PyPDFLoader(path)
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

    return all_chunks

# Example usage
documents = load_and_split_pdfs_from_folder("./policies")