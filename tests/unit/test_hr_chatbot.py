"""
Unit tests for HR Chatbot functionality.

This module provides comprehensive test coverage for the core
HR Chatbot functions and classes.
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from langchain_core.documents import Document

# Import modules to test
from config import Config, load_config, validate_config
from validation import InputValidator, ValidationError
from cache import QueryCache, EmbeddingCache
from chunking import SmartChunker, ChunkingConfig
from hr_chatbot import load_and_split_pdfs, build_vectorstore, answer_query


class TestConfig:
    """Test configuration management."""
    
    def test_load_config_with_api_key(self):
        """Test loading config with API key."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            config = load_config()
            assert config.openai_api_key == 'test-key'
            assert config.embedding_model == 'text-embedding-3-large'
    
    def test_load_config_without_api_key(self):
        """Test loading config without API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                load_config()
    
    def test_load_config_with_custom_values(self):
        """Test loading config with custom environment variables."""
        env_vars = {
            'OPENAI_API_KEY': 'test-key',
            'CHUNK_SIZE': '1000',
            'CHUNK_OVERLAP': '200',
            'LLM_MODEL': 'gpt-3.5-turbo'
        }
        with patch.dict(os.environ, env_vars):
            config = load_config()
            assert config.chunk_size == 1000
            assert config.chunk_overlap == 200
            assert config.llm_model == 'gpt-3.5-turbo'
    
    def test_validate_config_valid(self):
        """Test validating a valid configuration."""
        config = Config(
            openai_api_key='test-key',
            chunk_size=500,
            chunk_overlap=100
        )
        # Should not raise any exception
        validate_config(config)
    
    def test_validate_config_invalid_chunk_overlap(self):
        """Test validating config with invalid chunk overlap."""
        config = Config(
            openai_api_key='test-key',
            chunk_size=500,
            chunk_overlap=600  # Greater than chunk_size
        )
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            validate_config(config)


class TestInputValidator:
    """Test input validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = InputValidator()
    
    def test_validate_query_valid(self):
        """Test validating a valid query."""
        query = "What is the vacation policy?"
        result = self.validator.validate_query(query)
        assert result == query
    
    def test_validate_query_empty(self):
        """Test validating empty query."""
        with pytest.raises(ValidationError, match="Query cannot be empty"):
            self.validator.validate_query("")
    
    def test_validate_query_too_long(self):
        """Test validating overly long query."""
        long_query = "x" * 2000  # Longer than max_query_length
        with pytest.raises(ValidationError, match="Query too long"):
            self.validator.validate_query(long_query)
    
    def test_validate_query_dangerous_content(self):
        """Test validating query with dangerous content."""
        dangerous_query = "<script>alert('xss')</script>"
        with pytest.raises(ValidationError, match="potentially dangerous content"):
            self.validator.validate_query(dangerous_query)
    
    def test_validate_query_sql_injection(self):
        """Test validating query with SQL injection attempt."""
        sql_query = "'; DROP TABLE users; --"
        with pytest.raises(ValidationError, match="potentially malicious SQL patterns"):
            self.validator.validate_query(sql_query)
    
    def test_sanitize_string(self):
        """Test string sanitization."""
        input_str = "<script>alert('test')</script> & some text"
        result = self.validator.sanitize_string(input_str)
        assert "<script>" not in result
        assert "&lt;script&gt;" in result
        assert "&amp;" in result
    
    def test_validate_file_path_valid(self):
        """Test validating valid file path."""
        with tempfile.NamedTemporaryFile() as tmp:
            path = self.validator.validate_file_path(tmp.name)
            assert isinstance(path, Path)
    
    def test_validate_file_path_traversal(self):
        """Test validating path traversal attempt."""
        with pytest.raises(ValidationError, match="potential path traversal"):
            self.validator.validate_file_path("../../../etc/passwd")
    
    def test_validate_k_value_valid(self):
        """Test validating valid k value."""
        result = self.validator.validate_k_value(5)
        assert result == 5
    
    def test_validate_k_value_invalid(self):
        """Test validating invalid k value."""
        with pytest.raises(ValidationError, match="k must be positive"):
            self.validator.validate_k_value(0)
    
    def test_validate_model_name_valid(self):
        """Test validating valid model name."""
        result = self.validator.validate_model_name("gpt-4o")
        assert result == "gpt-4o"
    
    def test_validate_model_name_invalid_chars(self):
        """Test validating model name with invalid characters."""
        with pytest.raises(ValidationError, match="invalid characters"):
            self.validator.validate_model_name("gpt-4o@#$%")


class TestQueryCache:
    """Test query caching functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = QueryCache(max_size=10, ttl=3600)
    
    def test_cache_set_and_get(self):
        """Test setting and getting cache entries."""
        query = "test query"
        model = "gpt-4o"
        k = 4
        result = ("test answer", [])
        
        # Set cache entry
        self.cache.set(query, model, k, result)
        
        # Get cache entry
        cached_result = self.cache.get(query, model, k)
        assert cached_result == result
    
    def test_cache_miss(self):
        """Test cache miss scenario."""
        result = self.cache.get("non-existent query", "gpt-4o", 4)
        assert result is None
    
    def test_cache_clear(self):
        """Test clearing cache."""
        self.cache.set("test", "gpt-4o", 4, ("answer", []))
        self.cache.clear()
        result = self.cache.get("test", "gpt-4o", 4)
        assert result is None
    
    def test_cache_stats(self):
        """Test getting cache statistics."""
        stats = self.cache.get_stats()
        assert 'total_entries' in stats
        assert 'active_entries' in stats
        assert 'max_size' in stats


class TestEmbeddingCache:
    """Test embedding caching functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = EmbeddingCache(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_embedding_cache_set_and_get(self):
        """Test setting and getting embedding cache entries."""
        content = "test document content"
        model = "text-embedding-3-large"
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Set embedding
        self.cache.set_embedding(content, model, embedding)
        
        # Get embedding
        cached_embedding = self.cache.get_embedding(content, model)
        assert cached_embedding == embedding
    
    def test_embedding_cache_miss(self):
        """Test embedding cache miss."""
        result = self.cache.get_embedding("non-existent content", "test-model")
        assert result is None
    
    def test_embedding_cache_stats(self):
        """Test getting embedding cache statistics."""
        stats = self.cache.get_stats()
        assert 'total_embeddings' in stats
        assert 'estimated_size_mb' in stats


class TestSmartChunker:
    """Test smart chunking functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ChunkingConfig(chunk_size=200, chunk_overlap=50)
        self.chunker = SmartChunker(self.config)
    
    def test_detect_content_type_policy(self):
        """Test detecting policy content type."""
        policy_text = "This policy is mandatory and must be followed by all employees."
        content_type = self.chunker._detect_content_type(policy_text)
        assert content_type == "policy"
    
    def test_detect_content_type_procedure(self):
        """Test detecting procedure content type."""
        procedure_text = "1. First step\n2. Second step\n3. Third step"
        content_type = self.chunker._detect_content_type(procedure_text)
        assert content_type == "procedure"
    
    def test_chunk_document(self):
        """Test chunking a document."""
        doc = Document(
            page_content="This is a test document with some content that should be chunked appropriately.",
            metadata={"source": "test.pdf", "page": 1}
        )
        
        chunks = self.chunker.chunk_document(doc)
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)
        assert all('chunk_id' in chunk.metadata for chunk in chunks)
    
    def test_chunk_documents_multiple(self):
        """Test chunking multiple documents."""
        docs = [
            Document(
                page_content="First document content.",
                metadata={"source": "doc1.pdf"}
            ),
            Document(
                page_content="Second document content.",
                metadata={"source": "doc2.pdf"}
            )
        ]
        
        chunks = self.chunker.chunk_documents(docs)
        assert len(chunks) >= 2
        assert all(isinstance(chunk, Document) for chunk in chunks)


class TestHRChatbotFunctions:
    """Test main HR chatbot functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('hr_chatbot.PyPDFLoader')
    @patch('hr_chatbot.create_chunker')
    def test_load_and_split_pdfs(self, mock_chunker, mock_loader):
        """Test loading and splitting PDFs."""
        # Create a test PDF file
        test_pdf = os.path.join(self.temp_dir, "test.pdf")
        with open(test_pdf, 'w') as f:
            f.write("dummy pdf content")
        
        # Mock PDF loader
        mock_doc = Document(page_content="Test content", metadata={"source": test_pdf})
        mock_loader.return_value.load.return_value = [mock_doc]
        
        # Mock chunker
        mock_chunker_instance = Mock()
        mock_chunker_instance.chunk_documents.return_value = [mock_doc]
        mock_chunker.return_value = mock_chunker_instance
        
        # Test the function
        result = load_and_split_pdfs(self.temp_dir)
        
        assert len(result) == 1
        assert result[0].page_content == "Test content"
        mock_loader.assert_called_once()
        mock_chunker.assert_called_once()
    
    def test_load_and_split_pdfs_no_pdfs(self):
        """Test loading from directory with no PDFs."""
        with pytest.raises(ValueError, match="No PDF files found"):
            load_and_split_pdfs(self.temp_dir)
    
    def test_load_and_split_pdfs_nonexistent_dir(self):
        """Test loading from nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            load_and_split_pdfs("/nonexistent/directory")
    
    @patch('hr_chatbot.OpenAIEmbeddings')
    @patch('hr_chatbot.FAISS')
    def test_build_vectorstore(self, mock_faiss, mock_embeddings):
        """Test building vector store."""
        # Mock documents
        docs = [Document(page_content="Test", metadata={})]
        
        # Mock FAISS
        mock_vectorstore = Mock()
        mock_faiss.from_documents.return_value = mock_vectorstore
        
        # Test the function
        result = build_vectorstore(docs)
        
        assert result == mock_vectorstore
        mock_faiss.from_documents.assert_called_once()
        mock_vectorstore.save_local.assert_called_once()
    
    def test_build_vectorstore_empty_docs(self):
        """Test building vector store with empty documents."""
        with pytest.raises(ValueError, match="Documents list cannot be empty"):
            build_vectorstore([])
    
    @patch('hr_chatbot.get_query_cache')
    @patch('hr_chatbot.ChatOpenAI')
    @patch('hr_chatbot.RetrievalQA')
    def test_answer_query(self, mock_qa, mock_llm, mock_cache):
        """Test answering a query."""
        # Mock cache miss
        mock_cache.return_value.get.return_value = None
        
        # Mock vectorstore
        mock_vectorstore = Mock()
        mock_vectorstore.as_retriever.return_value = Mock()
        
        # Mock QA chain
        mock_qa_instance = Mock()
        mock_qa_instance.invoke.return_value = {
            'result': 'Test answer',
            'source_documents': []
        }
        mock_qa.from_chain_type.return_value = mock_qa_instance
        
        # Test the function
        answer, docs = answer_query("test question", mock_vectorstore)
        
        assert answer == 'Test answer'
        assert docs == []
        mock_llm.assert_called_once()
        mock_qa.from_chain_type.assert_called_once()
    
    def test_answer_query_empty_question(self):
        """Test answering empty question."""
        mock_vectorstore = Mock()
        with pytest.raises(ValidationError, match="Query cannot be empty"):
            answer_query("", mock_vectorstore)
    
    def test_answer_query_none_vectorstore(self):
        """Test answering query with None vectorstore."""
        with pytest.raises(ValueError, match="Vectorstore cannot be None"):
            answer_query("test question", None)


# Integration tests
class TestIntegration:
    """Integration tests for the HR Chatbot system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_config_validator_integration(self):
        """Test integration between config and validator."""
        config = load_config()
        validator = InputValidator()
        
        # Test that validator respects config limits
        long_query = "x" * (config.max_query_length + 1)
        with pytest.raises(ValidationError):
            validator.validate_query(long_query)
    
    def test_cache_chunker_integration(self):
        """Test integration between cache and chunker."""
        cache = QueryCache(max_size=5, ttl=3600)
        chunker = SmartChunker()
        
        # Test that they can work together
        doc = Document(page_content="Test content", metadata={})
        chunks = chunker.chunk_document(doc)
        
        # Cache should be able to store results
        cache.set("test", "gpt-4o", 4, ("answer", chunks))
        cached_result = cache.get("test", "gpt-4o", 4)
        
        assert cached_result is not None
        assert cached_result[1] == chunks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])