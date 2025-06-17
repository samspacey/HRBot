"""
hr_chatbot.py - Core functions for HR Policy Chatbot.

Dependencies:
    pip install langchain openai faiss-cpu tiktoken

Ensure OPENAI_API_KEY is set in your environment.
"""
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Ensure OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

def load_and_split_pdfs(
    folder_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List:
    """
    Load all PDF files from a folder and split into token-based chunks.
    Returns a list of LangChain Document objects, each with metadata.source set.
    """
    # Collect PDF paths
    pdf_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".pdf")
    ]
    # Initialize token-based splitter
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        # Tag source in metadata
        for doc in docs:
            doc.metadata["source"] = path
        # Split and collect
        splits = splitter.split_documents(docs)
        documents.extend(splits)
    return documents

def build_vectorstore(
    documents,
    embedding_model_name: str = "text-embedding-3-large",
    index_path: str = "faiss_index_hr",
) -> FAISS:
    """
    Build a FAISS vector store from documents and save locally.
    Returns the vectorstore instance.
    """
    embeddings = OpenAIEmbeddings(model=embedding_model_name)
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore

def load_vectorstore(
    index_path: str = "faiss_index_hr",
    embedding_model_name: str = "text-embedding-3-large",
) -> FAISS:
    """
    Load a FAISS vector store from local disk.
    """
    embeddings = OpenAIEmbeddings(model=embedding_model_name)
    # Allow deserializing our own index (pickle) since we trust our local files
    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore

def answer_query(
    question: str,
    vectorstore: FAISS,
    llm_model_name: str = "gpt-4o",
    k: int = 4,
) -> (str, List):
    """
    Use RetrievalQA to answer a question given a vectorstore.
    Returns the answer string and list of source Documents.
    """
    llm = ChatOpenAI(model=llm_model_name, temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    # Prompt: ground answers strictly in the given context
    prompt = PromptTemplate(
        template=(
            "You are a helpful HR assistant. Below are excerpts from the company's policies.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer concisely based ONLY on the provided context.\n"
            "If the information is not present, respond exactly with \"I don't know.\""
        ),
        input_variables=["context", "question"],
    )
    # Build a RetrievalQA chain using the customized prompt
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    # Use invoke() instead of deprecated direct call
    result = qa.invoke(question)
    answer = result["result"].strip()
    docs = result.get("source_documents", [])
    return answer, docs