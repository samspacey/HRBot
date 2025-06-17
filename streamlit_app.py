import os
import streamlit as st
from hr_chatbot import (
    load_and_split_pdfs,
    build_vectorstore,
    load_vectorstore,
    answer_query,
)

st.set_page_config(page_title="HR Policy Chatbot")
st.title("📄 HR Policy Chatbot")

# Sidebar for configuration
# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    folder = st.text_input("Policies folder path", "./policies")
    index_path = st.text_input("FAISS index path", "faiss_index_hr")
    rebuild = st.button("Rebuild Index")

# Automatically build or load the FAISS index on startup (or when rebuilding)
if "vectorstore" not in st.session_state or rebuild:
    with st.spinner("Building or loading FAISS index..."):
        # Try loading existing index unless forced to rebuild
        if os.path.exists(index_path) and not rebuild:
            try:
                vs = load_vectorstore(index_path)
            except Exception:
                docs = load_and_split_pdfs(folder)
                vs = build_vectorstore(docs, index_path=index_path)
        else:
            docs = load_and_split_pdfs(folder)
            vs = build_vectorstore(docs, index_path=index_path)
    st.session_state["vectorstore"] = vs
    st.success("FAISS index is ready!")

# Main chatbot UI
st.subheader("Ask a question about your HR policies:")
question = st.text_input("Your question:")
k = st.slider("Number of source documents (k)", 1, 10, 4)
ask_btn = st.button("Get Answer")

if ask_btn:
    if "vectorstore" not in st.session_state:
        st.warning("Please build or load the FAISS index first.")
    elif not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving answer..."):
            answer, docs = answer_query(question, st.session_state["vectorstore"], k=k)
        st.subheader("Answer")
        st.write(answer)
        st.subheader("Source Documents")
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "")
            st.markdown(f"**{i}. {source} (page {page})**")
            st.write(doc.page_content)