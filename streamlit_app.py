import os
import streamlit as st
from hr_chatbot import (
    load_and_split_pdfs,
    build_vectorstore,
    load_vectorstore,
    answer_query,
)

# Configure page
st.set_page_config(
    page_title="HR Policy Chatbot", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App header
st.title("🤖 HR Policy Chatbot")
st.markdown("*Ask questions about company HR policies using natural language*")

# Check for API key - Streamlit Cloud uses secrets
if "OPENAI_API_KEY" not in st.secrets:
    st.error("⚠️ OpenAI API key not found! Please add it to Streamlit secrets.")
    st.info("For Streamlit Cloud: Add OPENAI_API_KEY in the app's secrets section.")
    st.stop()

# Set environment variable for the app
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Check if policies folder exists
    folder = "./policies"
    index_path = "faiss_index_hr"
    
    if not os.path.exists(folder):
        st.error(f"📁 Policies folder '{folder}' not found!")
        st.info("Please ensure your HR policy PDFs are in the policies/ folder.")
        st.stop()
    
    # Show folder info
    pdf_files = [f for f in os.listdir(folder) if f.lower().endswith('.pdf')]
    st.success(f"📁 Found {len(pdf_files)} PDF files")
    
    # Rebuild option
    rebuild = st.button("🔄 Rebuild Index", help="Rebuild the search index from scratch")
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        k = st.slider("Source documents to retrieve", 1, 10, 4, 
                     help="Higher values provide more context but slower responses")

# Initialize vectorstore
@st.cache_resource
def initialize_vectorstore():
    """Initialize and cache the vectorstore"""
    if os.path.exists(index_path):
        try:
            return load_vectorstore(index_path)
        except Exception as e:
            st.warning(f"Failed to load existing index: {str(e)}")
    
    # Build new index
    with st.spinner("🔨 Building search index from PDF files..."):
        docs = load_and_split_pdfs(folder)
        vs = build_vectorstore(docs, index_path=index_path)
    return vs

# Load vectorstore (or rebuild if requested)
if rebuild:
    st.cache_resource.clear()

try:
    vectorstore = initialize_vectorstore()
    st.sidebar.success("✅ Search index ready!")
except Exception as e:
    st.error(f"❌ Failed to initialize search index: {str(e)}")
    st.stop()

# Main chatbot interface
st.markdown("---")
col1, col2 = st.columns([3, 1])

with col1:
    question = st.text_input(
        "💬 Ask a question about HR policies:",
        placeholder="e.g., What is the vacation policy?",
        help="Ask any question about your company's HR policies"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)  # Add some space
    ask_btn = st.button("🔍 Get Answer", type="primary", use_container_width=True)

# Process query
if ask_btn or question:
    if not question:
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("🤔 Thinking..."):
            try:
                answer, docs = answer_query(question, vectorstore, k=k)
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.markdown(answer)
                
                # Display sources
                if docs:
                    st.markdown("### 📚 Source Documents")
                    for i, doc in enumerate(docs, start=1):
                        source = doc.metadata.get("source", "unknown")
                        page = doc.metadata.get("page", "")
                        
                        with st.expander(f"📄 Source {i}: {os.path.basename(source)} (page {page})"):
                            st.write(doc.page_content)
                            
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit, LangChain, and OpenAI*")