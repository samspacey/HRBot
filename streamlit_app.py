import os
import streamlit as st
from document_chatbot import DocumentChatbot
from domain_config import load_current_domain_config, get_available_domains

# Load domain configuration
try:
    domain_config = load_current_domain_config()
    chatbot = DocumentChatbot(domain_config)
except Exception as e:
    st.error(f"❌ Error loading domain configuration: {str(e)}")
    st.info("Available domains: " + ", ".join(get_available_domains()))
    st.stop()

# Configure page with dynamic settings
st.set_page_config(
    page_title=domain_config.ui_page_title, 
    page_icon=domain_config.ui_page_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# App header with dynamic branding
st.title(domain_config.ui_title)
st.markdown(f"*{domain_config.description}*")

# Check for API key - Streamlit Cloud uses secrets
if "OPENAI_API_KEY" not in st.secrets:
    st.error(domain_config.messages.get("no_api_key", "⚠️ OpenAI API key not found!"))
    st.info(domain_config.messages.get("api_key_help", "For Streamlit Cloud: Add OPENAI_API_KEY in the app's secrets section."))
    st.stop()

# Set environment variable for the app
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Sidebar for configuration
with st.sidebar:
    st.header(domain_config.ui_sidebar_title)
    
    # Check if documents folder exists
    folder = domain_config.documents_folder
    index_path = domain_config.documents_index_path
    
    if not os.path.exists(folder):
        st.error(domain_config.messages["no_folder"].format(folder=folder))
        st.info(domain_config.messages["no_folder_help"].format(folder=folder))
        st.stop()
    
    # Show folder info
    document_files = [
        f for f in os.listdir(folder) 
        if any(f.lower().endswith(ext) for ext in domain_config.documents_file_types)
    ]
    st.success(domain_config.messages["found_files"].format(count=len(document_files)))
    
    # Rebuild option
    rebuild = st.button("🔄 Rebuild Index", help="Rebuild the search index from scratch")
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        k = st.slider("Source documents to retrieve", 1, domain_config.max_k, domain_config.default_k, 
                     help="Higher values provide more context but slower responses")

# Initialize vectorstore
@st.cache_resource
def initialize_vectorstore():
    """Initialize and cache the vectorstore"""
    if os.path.exists(index_path):
        try:
            return chatbot.load_vectorstore(index_path)
        except Exception as e:
            st.warning(f"Failed to load existing index: {str(e)}")
    
    # Build new index
    with st.spinner(f"🔨 Building search index from {domain_config.documents_folder_display_name.lower()}..."):
        docs = chatbot.load_and_split_documents(folder)
        vs = chatbot.build_vectorstore(docs, index_path=index_path)
    return vs

# Load vectorstore (or rebuild if requested)
if rebuild:
    st.cache_resource.clear()

try:
    vectorstore = initialize_vectorstore()
    st.sidebar.success(domain_config.messages["index_ready"])
except Exception as e:
    st.error(f"❌ Failed to initialize search index: {str(e)}")
    st.stop()

# Main chatbot interface
st.markdown("---")
col1, col2 = st.columns([3, 1])

with col1:
    question = st.text_input(
        f"💬 Ask a question about {domain_config.documents_folder_display_name.lower()}:",
        placeholder=domain_config.query_placeholder,
        help=domain_config.query_help_text
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)  # Add some space
    ask_btn = st.button(domain_config.query_button_text, type="primary", use_container_width=True)

# Process query
if ask_btn or question:
    if not question:
        st.warning(domain_config.messages["no_question"])
    else:
        with st.spinner("🤔 Thinking..."):
            try:
                answer, docs = chatbot.answer_query(question, vectorstore, k=k)
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.markdown(answer)
                
                # Display sources
                if docs:
                    st.markdown(f"### 📚 Source {domain_config.documents_folder_display_name}")
                    for i, doc in enumerate(docs, start=1):
                        source = doc.metadata.get("source", "unknown")
                        page = doc.metadata.get("page", "")
                        
                        with st.expander(f"📄 Source {i}: {os.path.basename(source)} (page {page})"):
                            st.write(doc.page_content)
                            
            except Exception as e:
                st.error(domain_config.messages["processing_error"].format(error=str(e)))

# Footer
st.markdown("---")
st.markdown(f"*{domain_config.ui_footer}*")