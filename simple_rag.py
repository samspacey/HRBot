"""
Minimal RAG utilities for a simple demo chatbot.

Functions:
- build_or_load_index: Load existing FAISS index or build from PDFs in a folder.
- answer: Retrieve with FAISS and generate an answer using OpenAI via LangChain.

Environment: requires OPENAI_API_KEY.
"""

import os
from typing import List, Tuple, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from config import get_config


def _load_pdfs(folder: str) -> List:
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".pdf")
    ]
    documents = []
    for path in files:
        loader = PyPDFLoader(path)
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = path
        documents.extend(docs)
    return documents


def _split_docs(docs: List, chunk_size: int, chunk_overlap: int) -> List:
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def build_or_load_index(folder: str, index_path: str, api_key: Optional[str] = None) -> FAISS:
    cfg = get_config()
    key = api_key or cfg.openai_api_key
    embeddings = OpenAIEmbeddings(model=cfg.embedding_model, openai_api_key=key)

    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    docs = _load_pdfs(folder)
    if not docs:
        raise ValueError(f"No PDF files found in {folder}")
    chunks = _split_docs(docs, cfg.chunk_size, cfg.chunk_overlap)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(index_path)
    return vs


def answer(question: str, vectorstore: FAISS, k: int, api_key: Optional[str] = None) -> Tuple[str, List]:
    cfg = get_config()
    key = api_key or cfg.openai_api_key
    llm = ChatOpenAI(model=cfg.llm_model, temperature=0, openai_api_key=key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant answering questions using provided context.\n"
            "Use only the context. If unsure, say you don't know.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        ),
        input_variables=["context", "question"],
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    result = qa.invoke(question)
    answer_text = result["result"].strip()
    docs = result.get("source_documents", [])
    return answer_text, docs
