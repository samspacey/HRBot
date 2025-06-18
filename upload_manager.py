"""
upload_manager.py - File upload and management for interactive document chatbot.

This module handles file uploads, validation, temporary storage, and cleanup
for the interactive document upload feature.
"""

import os
import uuid
import shutil
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import streamlit as st
from validation import get_validator, ValidationError

logger = logging.getLogger(__name__)

class UploadManager:
    """
    Manages file uploads, validation, and temporary storage for user sessions.
    """
    
    # Supported file types and their extensions
    SUPPORTED_EXTENSIONS = {
        '.pdf': 'PDF Document',
        '.txt': 'Text File',
        '.md': 'Markdown File',
        '.docx': 'Word Document',
    }
    
    # File size limits (in bytes)
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_TOTAL_SIZE = 200 * 1024 * 1024  # 200MB total
    MAX_FILES = 20
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize upload manager for a session.
        
        Args:
            session_id: Unique session identifier (generated if not provided)
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.validator = get_validator()
        self.temp_dir = self._create_session_directory()
        
        # Initialize session state if not exists
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = {}
        if 'session_id' not in st.session_state:
            st.session_state.session_id = self.session_id
            
    def _create_session_directory(self) -> Path:
        """Create temporary directory for session files."""
        temp_base = Path(tempfile.gettempdir()) / "chatbot_sessions"
        temp_base.mkdir(exist_ok=True)
        
        session_dir = temp_base / self.session_id
        session_dir.mkdir(exist_ok=True)
        
        logger.info(f"Created session directory: {session_dir}")
        return session_dir
    
    def validate_file(self, uploaded_file: Any) -> Tuple[bool, str]:
        """
        Validate uploaded file for security and format compliance.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file extension
            file_ext = Path(uploaded_file.name).suffix.lower()
            if file_ext not in self.SUPPORTED_EXTENSIONS:
                return False, f"Unsupported file type: {file_ext}. Supported: {', '.join(self.SUPPORTED_EXTENSIONS.keys())}"
            
            # Check file size
            if uploaded_file.size > self.MAX_FILE_SIZE:
                return False, f"File too large: {uploaded_file.size / (1024*1024):.1f}MB (max: {self.MAX_FILE_SIZE / (1024*1024):.1f}MB)"
            
            # Check total uploaded size
            total_size = sum(info['size'] for info in st.session_state.uploaded_files.values())
            if total_size + uploaded_file.size > self.MAX_TOTAL_SIZE:
                return False, f"Total upload size limit exceeded (max: {self.MAX_TOTAL_SIZE / (1024*1024):.1f}MB)"
            
            # Check number of files
            if len(st.session_state.uploaded_files) >= self.MAX_FILES:
                return False, f"Too many files (max: {self.MAX_FILES})"
            
            # Validate file name
            try:
                self.validator.validate_filename(uploaded_file.name)
            except ValidationError as e:
                return False, f"Invalid filename: {str(e)}"
            
            return True, "File is valid"
            
        except Exception as e:
            logger.error(f"Error validating file {uploaded_file.name}: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
    def save_uploaded_file(self, uploaded_file: Any) -> Tuple[bool, str, Optional[str]]:
        """
        Save uploaded file to temporary session directory.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (success, message, file_path)
        """
        try:
            # Validate file first
            is_valid, validation_msg = self.validate_file(uploaded_file)
            if not is_valid:
                return False, validation_msg, None
            
            # Generate unique filename to avoid conflicts
            file_id = str(uuid.uuid4())[:8]
            original_name = uploaded_file.name
            file_ext = Path(original_name).suffix.lower()
            safe_name = f"{file_id}_{original_name}"
            
            # Save file to session directory
            file_path = self.temp_dir / safe_name
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Store file info in session state
            file_info = {
                'id': file_id,
                'original_name': original_name,
                'safe_name': safe_name,
                'path': str(file_path),
                'size': uploaded_file.size,
                'type': self.SUPPORTED_EXTENSIONS[file_ext],
                'extension': file_ext
            }
            
            st.session_state.uploaded_files[file_id] = file_info
            
            logger.info(f"Saved file: {original_name} -> {file_path}")
            return True, f"Successfully uploaded {original_name}", str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving file {uploaded_file.name}: {str(e)}")
            return False, f"Error saving file: {str(e)}", None
    
    def remove_file(self, file_id: str) -> Tuple[bool, str]:
        """
        Remove uploaded file from session.
        
        Args:
            file_id: Unique file identifier
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if file_id not in st.session_state.uploaded_files:
                return False, "File not found"
            
            file_info = st.session_state.uploaded_files[file_id]
            file_path = Path(file_info['path'])
            
            # Remove physical file
            if file_path.exists():
                file_path.unlink()
            
            # Remove from session state
            del st.session_state.uploaded_files[file_id]
            
            logger.info(f"Removed file: {file_info['original_name']}")
            return True, f"Removed {file_info['original_name']}"
            
        except Exception as e:
            logger.error(f"Error removing file {file_id}: {str(e)}")
            return False, f"Error removing file: {str(e)}"
    
    def get_uploaded_files(self) -> Dict[str, Dict]:
        """Get list of uploaded files for current session."""
        return st.session_state.uploaded_files.copy()
    
    def get_file_paths(self) -> List[str]:
        """Get list of file paths for uploaded files."""
        return [info['path'] for info in st.session_state.uploaded_files.values()]
    
    def clear_all_files(self) -> Tuple[bool, str]:
        """
        Remove all uploaded files from session.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            file_count = len(st.session_state.uploaded_files)
            
            # Remove all physical files
            for file_info in st.session_state.uploaded_files.values():
                file_path = Path(file_info['path'])
                if file_path.exists():
                    file_path.unlink()
            
            # Clear session state
            st.session_state.uploaded_files = {}
            
            logger.info(f"Cleared {file_count} files from session")
            return True, f"Removed all {file_count} files"
            
        except Exception as e:
            logger.error(f"Error clearing files: {str(e)}")
            return False, f"Error clearing files: {str(e)}"
    
    def get_total_size(self) -> int:
        """Get total size of uploaded files in bytes."""
        return sum(info['size'] for info in st.session_state.uploaded_files.values())
    
    def get_file_stats(self) -> Dict[str, Any]:
        """Get statistics about uploaded files."""
        files = st.session_state.uploaded_files
        
        if not files:
            return {
                'count': 0,
                'total_size': 0,
                'total_size_mb': 0,
                'types': {}
            }
        
        total_size = self.get_total_size()
        type_counts = {}
        
        for file_info in files.values():
            file_type = file_info['type']
            type_counts[file_type] = type_counts.get(file_type, 0) + 1
        
        return {
            'count': len(files),
            'total_size': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'types': type_counts
        }
    
    def cleanup_session(self) -> bool:
        """
        Clean up session directory and files.
        
        Returns:
            True if cleanup successful
        """
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up session directory: {self.temp_dir}")
            return True
        except Exception as e:
            logger.error(f"Error cleaning up session: {str(e)}")
            return False
    
    @staticmethod
    def cleanup_old_sessions(max_age_hours: int = 24) -> int:
        """
        Clean up old session directories.
        
        Args:
            max_age_hours: Maximum age of sessions to keep
            
        Returns:
            Number of sessions cleaned up
        """
        try:
            temp_base = Path(tempfile.gettempdir()) / "chatbot_sessions"
            if not temp_base.exists():
                return 0
            
            import time
            current_time = time.time()
            cleanup_count = 0
            
            for session_dir in temp_base.iterdir():
                if session_dir.is_dir():
                    # Check directory age
                    dir_age = current_time - session_dir.stat().st_mtime
                    if dir_age > (max_age_hours * 3600):
                        shutil.rmtree(session_dir)
                        cleanup_count += 1
                        logger.info(f"Cleaned up old session: {session_dir.name}")
            
            return cleanup_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {str(e)}")
            return 0


def get_upload_manager(session_id: Optional[str] = None) -> UploadManager:
    """
    Get upload manager instance for current session.
    
    Args:
        session_id: Optional session identifier
        
    Returns:
        UploadManager instance
    """
    if session_id is None and 'session_id' in st.session_state:
        session_id = st.session_state.session_id
    
    return UploadManager(session_id)