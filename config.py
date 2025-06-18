"""
Configuration management for HR Chatbot.

This module provides centralized configuration management using environment variables
and default values for the HR Policy Chatbot application.
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration settings for the HR Chatbot application."""
    
    # OpenAI API settings
    openai_api_key: str
    embedding_model: str = "text-embedding-3-large"
    llm_model: str = "gpt-4o"
    
    # Document processing settings
    chunk_size: int = 500
    chunk_overlap: int = 100
    
    # Vector store settings
    index_path: str = "faiss_index_hr"
    policies_folder: str = "./policies"
    
    # Query settings
    default_k: int = 4
    max_k: int = 10
    
    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Performance settings
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour in seconds
    max_cache_size: int = 1000
    
    # Security settings
    max_query_length: int = 1000
    allowed_file_extensions: tuple = (".pdf",)


def load_config() -> Config:
    """
    Load configuration from environment variables with fallback to defaults.
    
    Returns:
        Config: Configuration object with all settings
        
    Raises:
        ValueError: If required environment variables are missing
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required. "
            "Please set it to your OpenAI API key."
        )
    
    return Config(
        # Required settings
        openai_api_key=openai_api_key,
        
        # Optional settings with defaults
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o"),
        
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
        
        index_path=os.getenv("INDEX_PATH", "faiss_index_hr"),
        policies_folder=os.getenv("POLICIES_FOLDER", "./policies"),
        
        default_k=int(os.getenv("DEFAULT_K", "4")),
        max_k=int(os.getenv("MAX_K", "10")),
        
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_format=os.getenv(
            "LOG_FORMAT", 
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ),
        
        cache_enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
        cache_ttl=int(os.getenv("CACHE_TTL", "3600")),
        max_cache_size=int(os.getenv("MAX_CACHE_SIZE", "1000")),
        
        max_query_length=int(os.getenv("MAX_QUERY_LENGTH", "1000")),
    )


def validate_config(config: Config) -> None:
    """
    Validate configuration settings.
    
    Args:
        config: Configuration object to validate
        
    Raises:
        ValueError: If configuration values are invalid
    """
    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
        
    if config.chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
        
    if config.chunk_overlap >= config.chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")
        
    if config.default_k <= 0:
        raise ValueError("default_k must be positive")
        
    if config.max_k <= 0:
        raise ValueError("max_k must be positive")
        
    if config.default_k > config.max_k:
        raise ValueError("default_k cannot be greater than max_k")
        
    if config.cache_ttl <= 0:
        raise ValueError("cache_ttl must be positive")
        
    if config.max_cache_size <= 0:
        raise ValueError("max_cache_size must be positive")
        
    if config.max_query_length <= 0:
        raise ValueError("max_query_length must be positive")


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        Config: The global configuration object
    """
    global _config
    if _config is None:
        _config = load_config()
        validate_config(_config)
    return _config


def reload_config() -> Config:
    """
    Reload configuration from environment variables.
    
    Returns:
        Config: The reloaded configuration object
    """
    global _config
    _config = load_config()
    validate_config(_config)
    return _config