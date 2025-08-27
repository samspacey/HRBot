import os
import streamlit as st

from config import get_config, reload_config
from simple_rag import build_or_load_index, answer


st.set_page_config(page_title="Document Chatbot (Demo)", page_icon="💬", layout="wide")
st.title("Document Chatbot (Demo)")
st.markdown("Ask questions about your PDF documents. This is a minimal proof of concept.")

# API key resolution
ui_key = None
with st.sidebar:
    with st.expander("API Key"):
        ui_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...", help="Overrides env/secrets for this session.")
        if ui_key:
            st.session_state["override_openai_key"] = ui_key

# Resolve key preference: UI override > env (no secrets/.env fallback)
api_key = st.session_state.get("override_openai_key") or os.environ.get("OPENAI_API_KEY")
if not api_key or api_key.lower().startswith("your_") or "your_openai_api_key_here" in api_key:
    st.error("Valid OPENAI_API_KEY not provided. Set it in the sidebar, env, or Streamlit secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# Ensure fresh config in case module cache held an old key
cfg = reload_config()
folder = cfg.policies_folder
index_path = cfg.index_path

with st.sidebar:
    st.header("Settings")
    if not os.path.exists(folder):
        st.error(f"Folder not found: {folder}")
        st.stop()
    pdfs = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    st.success(f"Found {len(pdfs)} PDF(s) in {folder}")
    rebuild = st.button("🔄 Rebuild Index")
    k = st.slider("Sources to retrieve", 1, 8, cfg.default_k)
    with st.expander("Debug"):
        source = "ui override" if st.session_state.get("override_openai_key") else ("env" if os.environ.get("OPENAI_API_KEY") else "none")
        masked = api_key[:7] + "..." if api_key else "missing"
        st.caption(f"Key source: {source}; detected: {masked}")


@st.cache_resource
def get_vectorstore(folder: str, index_path: str, api_key: str):
    return build_or_load_index(folder, index_path, api_key=api_key)


if rebuild:
    st.cache_resource.clear()

try:
    vectorstore = get_vectorstore(folder, index_path, api_key)
    st.sidebar.info("Index ready")
except Exception as e:
    st.error(f"Failed to initialize index: {e}")
    st.stop()

st.markdown("---")
col1, col2 = st.columns([3, 1])
with col1:
    question = st.text_input("💬 Ask a question:", placeholder="What is the vacation policy?")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    ask_btn = st.button("Ask", type="primary", use_container_width=True)

if ask_btn or question:
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("🤔 Thinking..."):
            try:
                ans, docs = answer(question, vectorstore, k=k, api_key=api_key)
                st.markdown("### 💡 Answer")
                st.write(ans)
                if docs:
                    st.markdown("### 📚 Sources")
                    for i, doc in enumerate(docs, start=1):
                        src = os.path.basename(doc.metadata.get("source", "unknown"))
                        page = doc.metadata.get("page", "")
                        with st.expander(f"📄 {i}. {src} (page {page})"):
                            st.write(doc.page_content)
            except Exception as e:
                st.error(f"Error processing query: {e}")

st.markdown("---")
st.caption("Simple RAG demo. PDFs from the policies folder.")
