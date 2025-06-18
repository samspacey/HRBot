"""
session_chatbot.py - Session-based chatbot instances for interactive document upload.

This module provides temporary chatbot instances that work with uploaded documents
in user sessions, without requiring pre-configured domain files.
"""

import os
import uuid
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import streamlit as st

from document_chatbot import DocumentChatbot
from domain_config import DomainConfig
from file_processors import get_document_processor
from upload_manager import get_upload_manager
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

logger = logging.getLogger(__name__)

@dataclass
class SessionConfig:
    """Configuration for a session-based chatbot."""
    name: str
    description: str
    icon: str
    session_id: str
    temp_dir: str
    index_path: str
    
    # Processing parameters
    chunk_size: int = 500
    chunk_overlap: int = 100
    default_k: int = 4
    max_k: int = 10
    
    # UI customization
    primary_color: str = "#1f77b4"
    background_color: str = "#ffffff"


class SessionChatbot:
    """
    Session-based chatbot that works with uploaded documents.
    """
    
    def __init__(self, session_config: Optional[SessionConfig] = None):
        """
        Initialize session chatbot.
        
        Args:
            session_config: Configuration for the session (created if not provided)
        """
        self.session_config = session_config or self._create_default_config()
        self.upload_manager = get_upload_manager(self.session_config.session_id)
        self.document_processor = get_document_processor()
        self.vectorstore = None
        self.is_indexed = False
        
        # Create domain config for compatibility with DocumentChatbot
        self.domain_config = self._create_domain_config()
        self.chatbot = DocumentChatbot(self.domain_config)
        
        logger.info(f"Initialized session chatbot: {self.session_config.name}")
    
    def _create_default_config(self) -> SessionConfig:
        """Create default session configuration."""
        session_id = st.session_state.get('session_id', str(uuid.uuid4()))
        
        return SessionConfig(
            name="My Document Assistant",
            description="Chat with your uploaded documents",
            icon="📚",
            session_id=session_id,
            temp_dir=str(Path(tempfile.gettempdir()) / "chatbot_sessions" / session_id),
            index_path=str(Path(tempfile.gettempdir()) / "chatbot_sessions" / session_id / "session_index")
        )
    
    def _create_domain_config(self) -> DomainConfig:
        """Create a temporary domain config for compatibility."""
        return DomainConfig(
            name=self.session_config.name,
            description=self.session_config.description,
            domain="session",
            ui_title=f"{self.session_config.icon} {self.session_config.name}",
            ui_page_title=self.session_config.name,
            ui_page_icon=self.session_config.icon,
            ui_sidebar_title="⚙️ Document Upload",
            ui_footer="Built with Streamlit, LangChain, and OpenAI",
            documents_folder=self.session_config.temp_dir,
            documents_folder_display_name="Uploaded Documents",
            documents_file_types=[".pdf", ".txt", ".md", ".docx"],
            documents_index_path=self.session_config.index_path,
            query_placeholder="Ask a question about your uploaded documents...",
            query_help_text="Upload documents and ask questions about their content",
            query_button_text="🔍 Search Documents",
            system_prompt="""You are a helpful document assistant. Below are excerpts from uploaded documents.

Context:
{context}

Question: {question}
Answer based ONLY on the provided context from the uploaded documents.
If the information is not present, respond exactly with "I don't know."
Be specific and cite document names when possible.""",
            chunk_size=self.session_config.chunk_size,
            chunk_overlap=self.session_config.chunk_overlap,
            default_k=self.session_config.default_k,
            max_k=self.session_config.max_k,
            messages={
                'no_folder': '📁 No documents uploaded yet!',
                'no_folder_help': 'Please upload documents using the sidebar.',
                'no_api_key': '⚠️ OpenAI API key not found! Please add it to Streamlit secrets.',
                'api_key_help': 'For Streamlit Cloud: Add OPENAI_API_KEY in the app\'s secrets section.',
                'no_question': '⚠️ Please enter a question.',
                'processing_error': '❌ Error processing query: {error}',
                'index_ready': '✅ Document index ready!',
                'found_files': '📁 Found {count} uploaded documents'
            }
        )
    
    def update_config(self, name: str = None, description: str = None, icon: str = None,
                     chunk_size: int = None, chunk_overlap: int = None) -> None:
        """
        Update session configuration.
        
        Args:
            name: New chatbot name
            description: New description
            icon: New icon
            chunk_size: New chunk size
            chunk_overlap: New chunk overlap
        """
        if name:
            self.session_config.name = name
            self.domain_config.name = name
            self.domain_config.ui_page_title = name
        
        if description:
            self.session_config.description = description
            self.domain_config.description = description
        
        if icon:
            self.session_config.icon = icon
            self.domain_config.ui_page_icon = icon
        
        if chunk_size:
            self.session_config.chunk_size = chunk_size
            self.domain_config.chunk_size = chunk_size
        
        if chunk_overlap:
            self.session_config.chunk_overlap = chunk_overlap
            self.domain_config.chunk_overlap = chunk_overlap
        
        # Update UI title
        self.domain_config.ui_title = f"{self.session_config.icon} {self.session_config.name}"
        
        logger.info(f"Updated session config: {self.session_config.name}")
    
    def has_documents(self) -> bool:
        """Check if session has uploaded documents."""
        uploaded_files = self.upload_manager.get_uploaded_files()
        return len(uploaded_files) > 0
    
    def get_document_count(self) -> int:
        """Get number of uploaded documents."""
        return len(self.upload_manager.get_uploaded_files())
    
    def build_index(self, progress_callback=None) -> Tuple[bool, str]:
        """
        Build search index from uploaded documents.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Check if we have documents
            if not self.has_documents():
                return False, "No documents uploaded. Please upload documents first."
            
            # Get file paths
            file_paths = self.upload_manager.get_file_paths()
            
            if progress_callback:
                progress_callback(0.1, "Processing uploaded documents...")
            
            # Process documents
            documents = self.document_processor.process_files(file_paths)
            
            if not documents:
                return False, "No content could be extracted from uploaded documents."
            
            if progress_callback:
                progress_callback(0.4, f"Splitting {len(documents)} documents into chunks...")
            
            # Split documents using the chatbot's chunking
            split_docs = []
            for doc in documents:
                chunks = self.chatbot._split_document(doc)
                split_docs.extend(chunks)
            
            if progress_callback:
                progress_callback(0.7, f"Building search index from {len(split_docs)} chunks...")
            
            # Create vectorstore
            self.vectorstore = self.chatbot.build_vectorstore(
                split_docs, 
                index_path=self.session_config.index_path
            )
            
            self.is_indexed = True
            
            if progress_callback:
                progress_callback(1.0, "Index built successfully!")
            
            logger.info(f"Built index for session {self.session_config.session_id}: {len(split_docs)} chunks")
            return True, f"Successfully indexed {len(documents)} documents ({len(split_docs)} chunks)"
            
        except Exception as e:
            error_msg = f"Error building index: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def load_index(self) -> Tuple[bool, str]:
        """
        Load existing index if available.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            if os.path.exists(self.session_config.index_path):
                self.vectorstore = self.chatbot.load_vectorstore(self.session_config.index_path)
                self.is_indexed = True
                return True, "Index loaded successfully"
            else:
                return False, "No existing index found"
                
        except Exception as e:
            error_msg = f"Error loading index: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def answer_query(self, question: str, k: int = None) -> Tuple[str, List[Document]]:
        """
        Answer a question using the indexed documents.
        
        Args:
            question: User question
            k: Number of source documents to retrieve
            
        Returns:
            Tuple of (answer, source_documents)
            
        Raises:
            ValueError: If index is not built or question is invalid
        """
        if not self.is_indexed or self.vectorstore is None:
            raise ValueError("Index not built. Please build the index first.")
        
        k = k or self.session_config.default_k
        
        return self.chatbot.answer_query(question, self.vectorstore, k=k)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about the current session."""
        file_stats = self.upload_manager.get_file_stats()
        
        return {
            'session_id': self.session_config.session_id,
            'name': self.session_config.name,
            'description': self.session_config.description,
            'has_documents': self.has_documents(),
            'document_count': file_stats['count'],
            'total_size_mb': file_stats['total_size_mb'],
            'file_types': file_stats['types'],
            'is_indexed': self.is_indexed,
            'index_path': self.session_config.index_path,
            'index_exists': os.path.exists(self.session_config.index_path) if self.session_config.index_path else False
        }
    
    def clear_session(self) -> Tuple[bool, str]:
        """
        Clear all session data including files and index.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            # Clear uploaded files
            success, msg = self.upload_manager.clear_all_files()
            if not success:
                return False, f"Error clearing files: {msg}"
            
            # Remove index
            if os.path.exists(self.session_config.index_path):
                import shutil
                if os.path.isdir(self.session_config.index_path):
                    shutil.rmtree(self.session_config.index_path)
                else:
                    os.remove(self.session_config.index_path)
            
            # Reset state
            self.vectorstore = None
            self.is_indexed = False
            
            # Clean up session directory
            self.upload_manager.cleanup_session()
            
            logger.info(f"Cleared session {self.session_config.session_id}")
            return True, "Session cleared successfully"
            
        except Exception as e:
            error_msg = f"Error clearing session: {str(e)}"
            logger.error(error_msg)
            return False, error_msg


def get_session_chatbot(session_config: Optional[SessionConfig] = None) -> SessionChatbot:
    """
    Get session chatbot instance.
    
    Args:
        session_config: Optional session configuration
        
    Returns:
        SessionChatbot instance
    """
    return SessionChatbot(session_config)


def create_session_config(name: str, description: str = None, icon: str = "📚", 
                         **kwargs) -> SessionConfig:
    """
    Create a session configuration.
    
    Args:
        name: Chatbot name
        description: Chatbot description
        icon: Chatbot icon
        **kwargs: Additional configuration parameters
        
    Returns:
        SessionConfig instance
    """
    session_id = st.session_state.get('session_id', str(uuid.uuid4()))
    description = description or f"Chat with documents in {name}"
    
    return SessionConfig(
        name=name,
        description=description,
        icon=icon,
        session_id=session_id,
        temp_dir=str(Path(tempfile.gettempdir()) / "chatbot_sessions" / session_id),
        index_path=str(Path(tempfile.gettempdir()) / "chatbot_sessions" / session_id / "session_index"),
        **kwargs
    )