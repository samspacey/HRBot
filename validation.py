"""
Input validation and sanitization for HR Chatbot.

This module provides comprehensive input validation and sanitization
to ensure data integrity and security.
"""

import re
import html
import logging
from typing import Optional, List, Any, Dict
from pathlib import Path
from config import get_config

config = get_config()
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class InputValidator:
    """
    Comprehensive input validator for HR Chatbot.
    """
    
    # Potentially dangerous patterns
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',                # JavaScript URLs
        r'on\w+\s*=',                 # Event handlers
        r'eval\s*\(',                 # eval() calls
        r'exec\s*\(',                 # exec() calls
        r'import\s+os',               # OS imports
        r'import\s+subprocess',       # Subprocess imports
        r'__import__',                # Dynamic imports
        r'getattr\s*\(',             # getattr calls
        r'setattr\s*\(',             # setattr calls
    ]
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r'(?i)(union\s+select)',
        r'(?i)(drop\s+table)',
        r'(?i)(delete\s+from)',
        r'(?i)(insert\s+into)',
        r'(?i)(update\s+\w+\s+set)',
        r'(?i)(or\s+1\s*=\s*1)',
        r'(?i)(and\s+1\s*=\s*1)',
        r'[\'"]\s*;\s*--',
    ]
    
    def __init__(self):
        """Initialize the validator."""
        self.compiled_dangerous = [re.compile(pattern, re.IGNORECASE) 
                                 for pattern in self.DANGEROUS_PATTERNS]
        self.compiled_sql = [re.compile(pattern) 
                           for pattern in self.SQL_INJECTION_PATTERNS]
    
    def sanitize_string(self, text: str) -> str:
        """
        Sanitize a string by removing/escaping dangerous content.
        
        Args:
            text: Input string to sanitize
            
        Returns:
            Sanitized string
        """
        if not isinstance(text, str):
            raise ValidationError(f"Expected string, got {type(text)}")
        
        # HTML escape
        sanitized = html.escape(text)
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    def validate_query(self, query: str) -> str:
        """
        Validate and sanitize a user query.
        
        Args:
            query: User's question/query
            
        Returns:
            Validated and sanitized query
            
        Raises:
            ValidationError: If query fails validation
        """
        if not query:
            raise ValidationError("Query cannot be empty")
        
        if not isinstance(query, str):
            raise ValidationError(f"Query must be a string, got {type(query)}")
        
        # Basic length check
        if len(query) > config.max_query_length:
            raise ValidationError(
                f"Query too long: {len(query)} characters "
                f"(max: {config.max_query_length})"
            )
        
        # Check for dangerous patterns
        for pattern in self.compiled_dangerous:
            if pattern.search(query):
                logger.warning(f"Dangerous pattern detected in query: {query[:100]}...")
                raise ValidationError("Query contains potentially dangerous content")
        
        # Check for SQL injection attempts
        for pattern in self.compiled_sql:
            if pattern.search(query):
                logger.warning(f"SQL injection attempt detected: {query[:100]}...")
                raise ValidationError("Query contains potentially malicious SQL patterns")
        
        # Sanitize the query
        sanitized_query = self.sanitize_string(query)
        
        # Final length check after sanitization
        if len(sanitized_query.strip()) == 0:
            raise ValidationError("Query cannot be empty after sanitization")
        
        if len(sanitized_query.strip()) < 3:
            raise ValidationError("Query too short (minimum 3 characters)")
        
        logger.debug(f"Query validated: {sanitized_query[:50]}...")
        return sanitized_query
    
    def validate_file_path(self, file_path: str) -> Path:
        """
        Validate a file path for security.
        
        Args:
            file_path: Path to validate
            
        Returns:
            Validated Path object
            
        Raises:
            ValidationError: If path is invalid or unsafe
        """
        if not file_path:
            raise ValidationError("File path cannot be empty")
        
        if not isinstance(file_path, str):
            raise ValidationError(f"File path must be a string, got {type(file_path)}")
        
        # Convert to Path object
        try:
            path = Path(file_path).resolve()
        except Exception as e:
            raise ValidationError(f"Invalid file path: {str(e)}")
        
        # Check for path traversal attempts
        if '..' in file_path:
            logger.warning(f"Potential path traversal attempt: {file_path}")
            raise ValidationError("Invalid file path: potential path traversal")
        
        # Allow absolute paths for common temp directories and current working directory
        if file_path.startswith('/'):
            allowed_prefixes = ('/tmp', '/var/tmp', str(Path.cwd()), '/private/var/folders', '/var/folders')
            if not any(file_path.startswith(prefix) for prefix in allowed_prefixes):
                logger.warning(f"Absolute path not in allowed directories: {file_path}")
                raise ValidationError("Invalid file path: absolute path not allowed")
        
        return path
    
    def validate_folder_path(self, folder_path: str) -> Path:
        """
        Validate a folder path.
        
        Args:
            folder_path: Folder path to validate
            
        Returns:
            Validated Path object
            
        Raises:
            ValidationError: If path is invalid
        """
        path = self.validate_file_path(folder_path)
        
        if not path.exists():
            raise ValidationError(f"Folder does not exist: {folder_path}")
        
        if not path.is_dir():
            raise ValidationError(f"Path is not a directory: {folder_path}")
        
        return path
    
    def validate_pdf_file(self, file_path: str) -> Path:
        """
        Validate a PDF file path.
        
        Args:
            file_path: PDF file path to validate
            
        Returns:
            Validated Path object
            
        Raises:
            ValidationError: If file is invalid
        """
        path = self.validate_file_path(file_path)
        
        if not path.exists():
            raise ValidationError(f"File does not exist: {file_path}")
        
        if not path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")
        
        if not path.suffix.lower() in config.allowed_file_extensions:
            raise ValidationError(
                f"Invalid file extension: {path.suffix}. "
                f"Allowed: {config.allowed_file_extensions}"
            )
        
        # Check file size (max 100MB)
        max_size = 100 * 1024 * 1024  # 100MB
        if path.stat().st_size > max_size:
            raise ValidationError(f"File too large: {path.stat().st_size} bytes (max: {max_size})")
        
        return path
    
    def validate_integer(self, value: Any, min_val: int = None, max_val: int = None, name: str = "value") -> int:
        """
        Validate an integer value.
        
        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            name: Name of the parameter for error messages
            
        Returns:
            Validated integer
            
        Raises:
            ValidationError: If value is invalid
        """
        if not isinstance(value, (int, str)):
            raise ValidationError(f"{name} must be an integer, got {type(value)}")
        
        try:
            int_value = int(value)
        except (ValueError, TypeError):
            raise ValidationError(f"{name} must be a valid integer")
        
        if min_val is not None and int_value < min_val:
            raise ValidationError(f"{name} must be >= {min_val}, got {int_value}")
        
        if max_val is not None and int_value > max_val:
            raise ValidationError(f"{name} must be <= {max_val}, got {int_value}")
        
        return int_value
    
    def validate_model_name(self, model_name: str) -> str:
        """
        Validate an AI model name.
        
        Args:
            model_name: Model name to validate
            
        Returns:
            Validated model name
            
        Raises:
            ValidationError: If model name is invalid
        """
        if not model_name:
            raise ValidationError("Model name cannot be empty")
        
        if not isinstance(model_name, str):
            raise ValidationError(f"Model name must be a string, got {type(model_name)}")
        
        # Basic format validation
        if not re.match(r'^[a-zA-Z0-9\-_.]+$', model_name):
            raise ValidationError("Model name contains invalid characters")
        
        if len(model_name) > 100:
            raise ValidationError(f"Model name too long: {len(model_name)} characters (max: 100)")
        
        return model_name.strip()
    
    def validate_k_value(self, k: int) -> int:
        """
        Validate the k parameter for document retrieval.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            Validated k value
            
        Raises:
            ValidationError: If k is invalid
        """
        k = self.validate_integer(k, min_val=1, max_val=config.max_k, name="k")
        return k
    
    def validate_chunk_params(self, chunk_size: int, chunk_overlap: int) -> tuple[int, int]:
        """
        Validate document chunking parameters.
        
        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            Tuple of validated (chunk_size, chunk_overlap)
            
        Raises:
            ValidationError: If parameters are invalid
        """
        chunk_size = self.validate_integer(
            chunk_size, min_val=100, max_val=4000, name="chunk_size"
        )
        chunk_overlap = self.validate_integer(
            chunk_overlap, min_val=0, max_val=chunk_size-1, name="chunk_overlap"
        )
        
        if chunk_overlap >= chunk_size:
            raise ValidationError("chunk_overlap must be less than chunk_size")
        
        return chunk_size, chunk_overlap


# Global validator instance
_validator: Optional[InputValidator] = None


def get_validator() -> InputValidator:
    """Get the global validator instance."""
    global _validator
    if _validator is None:
        _validator = InputValidator()
    return _validator


def validate_query(query: str) -> str:
    """Convenience function to validate a query."""
    return get_validator().validate_query(query)


def validate_file_path(file_path: str) -> Path:
    """Convenience function to validate a file path."""
    return get_validator().validate_file_path(file_path)


def validate_folder_path(folder_path: str) -> Path:
    """Convenience function to validate a folder path."""
    return get_validator().validate_folder_path(folder_path)