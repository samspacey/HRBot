"""
streamlit_interactive.py - Interactive document upload chatbot interface.

This is the enhanced Streamlit app that allows users to upload their own documents
and create custom chatbots on-the-fly.
"""

import os
import streamlit as st
from pathlib import Path
from typing import Dict, Any

# Import our modules
from session_chatbot import SessionChatbot, SessionConfig, create_session_config
from upload_manager import get_upload_manager
from file_processors import get_document_processor

def initialize_session():
    """Initialize session state variables."""
    if 'session_chatbot' not in st.session_state:
        st.session_state.session_chatbot = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'index_built' not in st.session_state:
        st.session_state.index_built = False

def render_sidebar():
    """Render the sidebar with upload interface and settings."""
    with st.sidebar:
        st.header("📚 Document Upload")
        
        # Chatbot configuration section
        with st.expander("🤖 Chatbot Configuration", expanded=True):
            chatbot_name = st.text_input(
                "Chatbot Name", 
                value="My Document Assistant",
                help="Give your chatbot a custom name"
            )
            
            chatbot_description = st.text_area(
                "Description",
                value="Chat with your uploaded documents",
                height=60,
                help="Describe what your chatbot does"
            )
            
            # Icon selection
            icon_options = ["📚", "🤖", "📄", "💼", "🏥", "⚖️", "🔧", "📊", "🎓", "🌟"]
            selected_icon = st.selectbox(
                "Icon", 
                options=icon_options,
                index=0,
                help="Choose an icon for your chatbot"
            )
            
            # Update session chatbot config
            if st.session_state.session_chatbot:
                st.session_state.session_chatbot.update_config(
                    name=chatbot_name,
                    description=chatbot_description,
                    icon=selected_icon
                )
            else:
                # Create new session chatbot
                config = create_session_config(
                    name=chatbot_name,
                    description=chatbot_description,
                    icon=selected_icon
                )
                st.session_state.session_chatbot = SessionChatbot(config)
        
        # File upload section
        st.markdown("---")
        st.subheader("📁 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt', 'md', 'docx'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, Markdown, or Word documents"
        )
        
        # Process uploaded files
        upload_manager = get_upload_manager()
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Check if file is already uploaded
                existing_files = upload_manager.get_uploaded_files()
                file_already_exists = any(
                    info['original_name'] == uploaded_file.name 
                    for info in existing_files.values()
                )
                
                if not file_already_exists:
                    success, message, file_path = upload_manager.save_uploaded_file(uploaded_file)
                    if success:
                        st.success(f"✅ {message}")
                        st.session_state.index_built = False  # Need to rebuild index
                    else:
                        st.error(f"❌ {message}")
        
        # Display uploaded files
        uploaded_files = upload_manager.get_uploaded_files()
        if uploaded_files:
            st.markdown("### 📋 Uploaded Files")
            
            for file_id, file_info in uploaded_files.items():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    file_size_mb = file_info['size'] / (1024 * 1024)
                    st.write(f"📄 {file_info['original_name']}")
                    st.caption(f"{file_info['type']} • {file_size_mb:.1f} MB")
                
                with col2:
                    if st.button("🗑️", key=f"delete_{file_id}", help="Remove file"):
                        success, message = upload_manager.remove_file(file_id)
                        if success:
                            st.success(message)
                            st.session_state.index_built = False
                            st.rerun()
                        else:
                            st.error(message)
            
            # File statistics
            stats = upload_manager.get_file_stats()
            st.info(f"**{stats['count']} files** • **{stats['total_size_mb']} MB** total")
            
            # Clear all files button
            if st.button("🗑️ Clear All Files", type="secondary"):
                success, message = upload_manager.clear_all_files()
                if success:
                    st.success(message)
                    st.session_state.index_built = False
                    st.rerun()
                else:
                    st.error(message)
        
        else:
            st.info("No files uploaded yet. Upload documents to get started!")
        
        # Index building section
        st.markdown("---")
        st.subheader("🔍 Search Index")
        
        chatbot = st.session_state.session_chatbot
        if chatbot and chatbot.has_documents():
            if not st.session_state.index_built:
                if st.button("🔨 Build Search Index", type="primary"):
                    with st.spinner("Building search index..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        def progress_callback(progress, message):
                            progress_bar.progress(progress)
                            status_text.text(message)
                        
                        success, message = chatbot.build_index(progress_callback)
                        
                        if success:
                            st.success(message)
                            st.session_state.index_built = True
                            st.rerun()
                        else:
                            st.error(message)
            else:
                st.success("✅ Search index ready!")
                if st.button("🔄 Rebuild Index"):
                    st.session_state.index_built = False
                    st.rerun()
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            if chatbot:
                k = st.slider(
                    "Source documents to retrieve",
                    min_value=1,
                    max_value=chatbot.session_config.max_k,
                    value=chatbot.session_config.default_k,
                    help="Higher values provide more context but slower responses"
                )
                
                chunk_size = st.number_input(
                    "Chunk size",
                    min_value=100,
                    max_value=2000,
                    value=chatbot.session_config.chunk_size,
                    step=50,
                    help="Maximum tokens per document chunk"
                )
                
                chunk_overlap = st.number_input(
                    "Chunk overlap",
                    min_value=0,
                    max_value=500,
                    value=chatbot.session_config.chunk_overlap,
                    step=25,
                    help="Overlap between consecutive chunks"
                )
                
                # Update settings
                chatbot.update_config(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                # Store k value for queries
                st.session_state.k_value = k
            else:
                st.info("Upload documents to configure settings")

def render_main_interface():
    """Render the main chat interface."""
    chatbot = st.session_state.session_chatbot
    
    if not chatbot:
        st.title("📚 Interactive Document Chatbot")
        st.markdown("**Upload your documents and start chatting with them instantly!**")
        
        st.markdown("""
        ### 🚀 How to get started:
        1. **Configure your chatbot** in the sidebar (name, description, icon)
        2. **Upload documents** (PDF, TXT, MD, DOCX files)
        3. **Build the search index** to enable querying
        4. **Start asking questions** about your documents!
        
        ### ✨ Features:
        - **Multi-format support**: PDF, Text, Markdown, Word documents
        - **Smart chunking**: Optimized document processing for better answers
        - **Real-time indexing**: Build search indexes from your uploads
        - **Session-based**: Each session maintains its own document collection
        - **Customizable**: Configure chunk sizes, retrieval settings, and more
        """)
        return
    
    # Display chatbot header
    st.title(f"{chatbot.session_config.icon} {chatbot.session_config.name}")
    st.markdown(f"*{chatbot.session_config.description}*")
    
    # Check for API key
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("⚠️ OpenAI API key not found! Please add it to Streamlit secrets.")
        st.info("For Streamlit Cloud: Add OPENAI_API_KEY in the app's secrets section.")
        st.stop()
    
    # Set environment variable for the app
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    
    # Check if we have documents and index
    if not chatbot.has_documents():
        st.warning("📁 No documents uploaded yet!")
        st.info("👈 Upload documents using the sidebar to get started.")
        return
    
    if not st.session_state.index_built:
        st.warning("🔍 Search index not built yet!")
        st.info("👈 Build the search index using the sidebar to enable querying.")
        return
    
    # Main chat interface
    st.markdown("---")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 💬 Chat History")
        for i, (question, answer, sources) in enumerate(st.session_state.chat_history):
            with st.expander(f"Q: {question[:100]}{'...' if len(question) > 100 else ''}", expanded=(i == len(st.session_state.chat_history) - 1)):
                st.markdown("**Question:**")
                st.write(question)
                st.markdown("**Answer:**")
                st.write(answer)
                
                if sources:
                    st.markdown("**Sources:**")
                    for j, doc in enumerate(sources, 1):
                        source = doc.metadata.get("filename", "Unknown")
                        page = doc.metadata.get("page", "")
                        st.caption(f"📄 {j}. {source} {f'(page {page})' if page else ''}")
        
        st.markdown("---")
    
    # Query interface
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_input(
            "💬 Ask a question about your documents:",
            placeholder="e.g., What are the main topics discussed in the documents?",
            key="question_input"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add some space
        ask_btn = st.button("🔍 Ask", type="primary", use_container_width=True)
    
    # Process query
    if ask_btn or question:
        if not question:
            st.warning("⚠️ Please enter a question.")
        else:
            with st.spinner("🤔 Thinking..."):
                try:
                    k_value = st.session_state.get('k_value', chatbot.session_config.default_k)
                    answer, docs = chatbot.answer_query(question, k=k_value)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((question, answer, docs))
                    
                    # Display answer
                    st.markdown("### 💡 Answer")
                    st.markdown(answer)
                    
                    # Display sources
                    if docs:
                        st.markdown("### 📚 Source Documents")
                        for i, doc in enumerate(docs, start=1):
                            filename = doc.metadata.get("filename", "Unknown")
                            page = doc.metadata.get("page", "")
                            
                            with st.expander(f"📄 Source {i}: {filename} {f'(page {page})' if page else ''}"):
                                st.write(doc.page_content)
                    
                    # Clear the input
                    st.session_state.question_input = ""
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
    
    # Session statistics
    with st.expander("📊 Session Statistics"):
        stats = chatbot.get_session_stats()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Documents", stats['document_count'])
        
        with col2:
            st.metric("Total Size (MB)", f"{stats['total_size_mb']:.1f}")
        
        with col3:
            st.metric("Index Status", "✅ Ready" if stats['is_indexed'] else "❌ Not Built")
        
        st.write("**File Types:**", ", ".join(stats['file_types'].keys()) if stats['file_types'] else "None")
        st.write("**Session ID:**", stats['session_id'][:8] + "...")

def main():
    """Main application function."""
    # Configure page
    st.set_page_config(
        page_title="Interactive Document Chatbot",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session
    initialize_session()
    
    # Render interface
    render_sidebar()
    render_main_interface()
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit, LangChain, and OpenAI*")

if __name__ == "__main__":
    main()