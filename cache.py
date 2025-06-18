"""
Caching system for HR Chatbot.

This module provides caching functionality for embeddings and query results
to improve performance and reduce API calls.
"""

import hashlib
import json
import pickle
import time
from typing import Any, Optional, Dict, List, Tuple
from pathlib import Path
import logging
from config import get_config

config = get_config()
logger = logging.getLogger(__name__)


class QueryCache:
    """
    Simple in-memory cache for query results with TTL support.
    """
    
    def __init__(self, max_size: int = None, ttl: int = None):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of items to cache
            ttl: Time-to-live in seconds
        """
        self.max_size = max_size or config.max_cache_size
        self.ttl = ttl or config.cache_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        
    def _generate_key(self, query: str, model: str, k: int) -> str:
        """Generate a cache key from query parameters."""
        key_data = f"{query}:{model}:{k}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if a cache entry is expired."""
        return time.time() - timestamp > self.ttl
    
    def _evict_expired(self) -> None:
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if self._is_expired(entry['timestamp'])
        ]
        
        for key in expired_keys:
            del self.cache[key]
            del self.access_times[key]
            
        if expired_keys:
            logger.info(f"Evicted {len(expired_keys)} expired cache entries")
    
    def _evict_lru(self) -> None:
        """Remove least recently used entries if cache is full."""
        if len(self.cache) >= self.max_size:
            # Sort by access time and remove oldest
            sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
            keys_to_remove = [key for key, _ in sorted_keys[:len(sorted_keys)//4]]  # Remove 25%
            
            for key in keys_to_remove:
                del self.cache[key]
                del self.access_times[key]
                
            logger.info(f"Evicted {len(keys_to_remove)} LRU cache entries")
    
    def get(self, query: str, model: str, k: int) -> Optional[Tuple[str, List]]:
        """
        Get cached result for a query.
        
        Args:
            query: The user's question
            model: The LLM model name
            k: Number of documents retrieved
            
        Returns:
            Cached result tuple (answer, documents) or None if not found
        """
        if not config.cache_enabled:
            return None
            
        self._evict_expired()
        
        key = self._generate_key(query, model, k)
        
        if key in self.cache:
            entry = self.cache[key]
            if not self._is_expired(entry['timestamp']):
                self.access_times[key] = time.time()  # Update access time
                logger.info(f"Cache hit for query: {query[:50]}...")
                return entry['result']
            else:
                # Remove expired entry
                del self.cache[key]
                del self.access_times[key]
        
        logger.debug(f"Cache miss for query: {query[:50]}...")
        return None
    
    def set(self, query: str, model: str, k: int, result: Tuple[str, List]) -> None:
        """
        Cache a query result.
        
        Args:
            query: The user's question
            model: The LLM model name
            k: Number of documents retrieved
            result: The result tuple (answer, documents)
        """
        if not config.cache_enabled:
            return
            
        self._evict_expired()
        self._evict_lru()
        
        key = self._generate_key(query, model, k)
        current_time = time.time()
        
        self.cache[key] = {
            'result': result,
            'timestamp': current_time
        }
        self.access_times[key] = current_time
        
        logger.info(f"Cached result for query: {query[:50]}...")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_times.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        active_entries = sum(
            1 for entry in self.cache.values()
            if not self._is_expired(entry['timestamp'])
        )
        
        return {
            'total_entries': len(self.cache),
            'active_entries': active_entries,
            'expired_entries': len(self.cache) - active_entries,
            'max_size': self.max_size,
            'ttl': self.ttl,
            'cache_enabled': config.cache_enabled
        }


class EmbeddingCache:
    """
    Persistent cache for document embeddings to avoid recomputing.
    """
    
    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize the embedding cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "embeddings.pkl"
        self.metadata_file = self.cache_dir / "metadata.json"
        
        self.embeddings_cache: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        
        self._load_cache()
    
    def _generate_document_key(self, content: str, model: str) -> str:
        """Generate a unique key for a document."""
        key_data = f"{content}:{model}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                    
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                    
            logger.info(f"Loaded {len(self.embeddings_cache)} embeddings from cache")
            
        except Exception as e:
            logger.error(f"Error loading embedding cache: {str(e)}")
            self.embeddings_cache = {}
            self.metadata = {}
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
                
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
                
            logger.info(f"Saved {len(self.embeddings_cache)} embeddings to cache")
            
        except Exception as e:
            logger.error(f"Error saving embedding cache: {str(e)}")
    
    def get_embedding(self, content: str, model: str) -> Optional[List[float]]:
        """
        Get cached embedding for content.
        
        Args:
            content: Document content
            model: Embedding model name
            
        Returns:
            Cached embedding vector or None if not found
        """
        if not config.cache_enabled:
            return None
            
        key = self._generate_document_key(content, model)
        
        if key in self.embeddings_cache:
            logger.debug(f"Embedding cache hit for content: {content[:50]}...")
            return self.embeddings_cache[key]
            
        return None
    
    def set_embedding(self, content: str, model: str, embedding: List[float]) -> None:
        """
        Cache an embedding.
        
        Args:
            content: Document content
            model: Embedding model name
            embedding: Embedding vector
        """
        if not config.cache_enabled:
            return
            
        key = self._generate_document_key(content, model)
        self.embeddings_cache[key] = embedding
        
        # Update metadata
        self.metadata[key] = {
            'model': model,
            'timestamp': time.time(),
            'content_length': len(content)
        }
        
        logger.debug(f"Cached embedding for content: {content[:50]}...")
        
        # Periodically save to disk
        if len(self.embeddings_cache) % 100 == 0:
            self._save_cache()
    
    def clear(self) -> None:
        """Clear embedding cache."""
        self.embeddings_cache.clear()
        self.metadata.clear()
        
        # Remove cache files
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()
        except Exception as e:
            logger.error(f"Error clearing cache files: {str(e)}")
            
        logger.info("Embedding cache cleared")
    
    def save(self) -> None:
        """Manually save cache to disk."""
        self._save_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        total_size = sum(
            len(emb) * 4 for emb in self.embeddings_cache.values()  # Assume 4 bytes per float
        ) if self.embeddings_cache else 0
        
        return {
            'total_embeddings': len(self.embeddings_cache),
            'estimated_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir),
            'cache_enabled': config.cache_enabled
        }


# Global cache instances
_query_cache: Optional[QueryCache] = None
_embedding_cache: Optional[EmbeddingCache] = None


def get_query_cache() -> QueryCache:
    """Get the global query cache instance."""
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryCache()
    return _query_cache


def get_embedding_cache() -> EmbeddingCache:
    """Get the global embedding cache instance."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache