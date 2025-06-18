"""
Service layer for HR Chatbot with dependency injection.

This module provides a clean service architecture with dependency injection
to improve modularity, testability, and maintainability.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Protocol, runtime_checkable
from dataclasses import dataclass
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from config import Config, get_config
from cache import QueryCache, EmbeddingCache
from validation import InputValidator
from chunking import SmartChunker

logger = logging.getLogger(__name__)


# Protocols for dependency injection
@runtime_checkable
class EmbeddingService(Protocol):
    """Protocol for embedding services."""
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        ...
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        ...


@runtime_checkable
class LLMService(Protocol):
    """Protocol for language model services."""
    
    async def generate_response(self, prompt: str, context: str) -> str:
        """Generate a response given prompt and context."""
        ...


@runtime_checkable
class VectorStoreService(Protocol):
    """Protocol for vector store services."""
    
    async def create_store(self, documents: List[Document]) -> FAISS:
        """Create a vector store from documents."""
        ...
    
    async def load_store(self, path: str) -> FAISS:
        """Load a vector store from disk."""
        ...
    
    async def search(self, store: FAISS, query: str, k: int) -> List[Document]:
        """Search for similar documents."""
        ...


@runtime_checkable
class DocumentProcessor(Protocol):
    """Protocol for document processing services."""
    
    async def load_documents(self, folder_path: str) -> List[Document]:
        """Load documents from a folder."""
        ...
    
    async def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents into smaller pieces."""
        ...


@runtime_checkable
class CacheService(Protocol):
    """Protocol for caching services."""
    
    async def get_cached_response(self, query: str, model: str, k: int) -> Optional[Tuple[str, List[Document]]]:
        """Get cached response for a query."""
        ...
    
    async def cache_response(self, query: str, model: str, k: int, response: Tuple[str, List[Document]]) -> None:
        """Cache a response."""
        ...


# Concrete implementations
class OpenAIEmbeddingService:
    """OpenAI embedding service implementation."""
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            openai_api_key=config.openai_api_key
        )
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        try:
            return await self.embeddings.aembed_documents(texts)
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            raise
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        try:
            return await self.embeddings.aembed_query(text)
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise


class OpenAILLMService:
    """OpenAI language model service implementation."""
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=0,
            openai_api_key=config.openai_api_key
        )
    
    async def generate_response(self, prompt: str, context: str) -> str:
        """Generate a response given prompt and context."""
        try:
            formatted_prompt = prompt.format(context=context)
            response = await self.llm.ainvoke(formatted_prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise


class FAISSVectorStoreService:
    """FAISS vector store service implementation."""
    
    def __init__(self, embedding_service: EmbeddingService):
        """Initialize with embedding service."""
        self.embedding_service = embedding_service
    
    async def create_store(self, documents: List[Document]) -> FAISS:
        """Create a vector store from documents."""
        try:
            # Extract texts
            texts = [doc.page_content for doc in documents]
            
            # Get embeddings
            embeddings = await self.embedding_service.embed_documents(texts)
            
            # Create FAISS store
            # Note: This is a simplified version. Real implementation would need
            # to handle the FAISS creation more carefully
            from langchain_community.vectorstores import FAISS
            from langchain_openai import OpenAIEmbeddings
            
            # Use synchronous version for now as FAISS doesn't have async support
            embeddings_obj = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(documents, embeddings_obj)
            
            return vectorstore
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    async def load_store(self, path: str) -> FAISS:
        """Load a vector store from disk."""
        try:
            from langchain_openai import OpenAIEmbeddings
            
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.load_local(
                path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            return vectorstore
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise
    
    async def search(self, store: FAISS, query: str, k: int) -> List[Document]:
        """Search for similar documents."""
        try:
            docs = store.similarity_search(query, k=k)
            return docs
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise


class StandardDocumentProcessor:
    """Standard document processor implementation."""
    
    def __init__(self, validator: InputValidator, chunker: SmartChunker):
        """Initialize with validator and chunker."""
        self.validator = validator
        self.chunker = chunker
    
    async def load_documents(self, folder_path: str) -> List[Document]:
        """Load documents from a folder."""
        try:
            # Validate folder path
            validated_folder = self.validator.validate_folder_path(folder_path)
            
            # Load PDFs (simplified - real implementation would be async)
            from hr_chatbot import load_and_split_pdfs
            return load_and_split_pdfs(str(validated_folder))
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise
    
    async def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents into smaller pieces."""
        try:
            return self.chunker.chunk_documents(documents)
        except Exception as e:
            logger.error(f"Error chunking documents: {str(e)}")
            raise


class StandardCacheService:
    """Standard cache service implementation."""
    
    def __init__(self, query_cache: QueryCache):
        """Initialize with query cache."""
        self.query_cache = query_cache
    
    async def get_cached_response(
        self, 
        query: str, 
        model: str, 
        k: int
    ) -> Optional[Tuple[str, List[Document]]]:
        """Get cached response for a query."""
        return self.query_cache.get(query, model, k)
    
    async def cache_response(
        self, 
        query: str, 
        model: str, 
        k: int, 
        response: Tuple[str, List[Document]]
    ) -> None:
        """Cache a response."""
        self.query_cache.set(query, model, k, response)


# Service container for dependency injection
@dataclass
class ServiceContainer:
    """Container for all services with dependency injection."""
    
    config: Config
    embedding_service: EmbeddingService
    llm_service: LLMService
    vector_store_service: VectorStoreService
    document_processor: DocumentProcessor
    cache_service: CacheService
    validator: InputValidator


class ServiceFactory:
    """Factory for creating service instances."""
    
    @staticmethod
    def create_default_container() -> ServiceContainer:
        """Create a default service container with standard implementations."""
        config = get_config()
        
        # Create services
        embedding_service = OpenAIEmbeddingService(config)
        llm_service = OpenAILLMService(config)
        vector_store_service = FAISSVectorStoreService(embedding_service)
        
        # Create dependencies for document processor
        from validation import get_validator
        from chunking import create_chunker
        from cache import get_query_cache
        
        validator = get_validator()
        chunker = create_chunker(strategy="smart")
        query_cache = get_query_cache()
        
        document_processor = StandardDocumentProcessor(validator, chunker)
        cache_service = StandardCacheService(query_cache)
        
        return ServiceContainer(
            config=config,
            embedding_service=embedding_service,
            llm_service=llm_service,
            vector_store_service=vector_store_service,
            document_processor=document_processor,
            cache_service=cache_service,
            validator=validator
        )
    
    @staticmethod
    def create_test_container(
        embedding_service: Optional[EmbeddingService] = None,
        llm_service: Optional[LLMService] = None,
        vector_store_service: Optional[VectorStoreService] = None,
        document_processor: Optional[DocumentProcessor] = None,
        cache_service: Optional[CacheService] = None
    ) -> ServiceContainer:
        """Create a test service container with mock services."""
        config = get_config()
        
        # Use provided services or create defaults
        embedding_service = embedding_service or OpenAIEmbeddingService(config)
        llm_service = llm_service or OpenAILLMService(config)
        vector_store_service = vector_store_service or FAISSVectorStoreService(embedding_service)
        
        if document_processor is None:
            from validation import get_validator
            from chunking import create_chunker
            validator = get_validator()
            chunker = create_chunker(strategy="smart")
            document_processor = StandardDocumentProcessor(validator, chunker)
        
        if cache_service is None:
            from cache import get_query_cache
            query_cache = get_query_cache()
            cache_service = StandardCacheService(query_cache)
        
        from validation import get_validator
        validator = get_validator()
        
        return ServiceContainer(
            config=config,
            embedding_service=embedding_service,
            llm_service=llm_service,
            vector_store_service=vector_store_service,
            document_processor=document_processor,
            cache_service=cache_service,
            validator=validator
        )


# Main HR service class
class HRChatbotService:
    """Main HR Chatbot service with dependency injection."""
    
    def __init__(self, container: ServiceContainer):
        """Initialize with service container."""
        self.container = container
        self.config = container.config
        self.embedding_service = container.embedding_service
        self.llm_service = container.llm_service
        self.vector_store_service = container.vector_store_service
        self.document_processor = container.document_processor
        self.cache_service = container.cache_service
        self.validator = container.validator
    
    async def build_index(self, folder_path: Optional[str] = None) -> FAISS:
        """Build search index from documents."""
        folder_path = folder_path or self.config.policies_folder
        
        try:
            # Load and process documents
            logger.info(f"Loading documents from {folder_path}")
            documents = await self.document_processor.load_documents(folder_path)
            
            if not documents:
                raise ValueError("No documents found to index")
            
            # Chunk documents
            logger.info("Chunking documents")
            chunked_docs = await self.document_processor.chunk_documents(documents)
            
            # Create vector store
            logger.info("Creating vector store")
            vectorstore = await self.vector_store_service.create_store(chunked_docs)
            
            logger.info(f"Index built successfully with {len(chunked_docs)} chunks")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error building index: {str(e)}")
            raise
    
    async def load_index(self, index_path: Optional[str] = None) -> FAISS:
        """Load existing search index."""
        index_path = index_path or self.config.index_path
        
        try:
            logger.info(f"Loading index from {index_path}")
            vectorstore = await self.vector_store_service.load_store(index_path)
            logger.info("Index loaded successfully")
            return vectorstore
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise
    
    async def answer_query(
        self, 
        question: str, 
        vectorstore: FAISS, 
        k: Optional[int] = None
    ) -> Tuple[str, List[Document]]:
        """Answer a user query."""
        k = k or self.config.default_k
        
        try:
            # Validate inputs
            question = self.validator.validate_query(question)
            k = self.validator.validate_k_value(k)
            
            logger.info(f"Processing query: {question[:100]}...")
            
            # Check cache
            cached_result = await self.cache_service.get_cached_response(
                question, self.config.llm_model, k
            )
            if cached_result is not None:
                logger.info("Returning cached result")
                return cached_result
            
            # Search for relevant documents
            docs = await self.vector_store_service.search(vectorstore, question, k)
            
            # Prepare context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate response
            prompt = (
                "You are a helpful HR assistant. Below are excerpts from the company's policies.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n"
                "Answer concisely based ONLY on the provided context.\n"
                "If the information is not present, respond exactly with \"I don't know.\""
            )
            
            full_prompt = prompt.format(context=context, question=question)
            answer = await self.llm_service.generate_response(full_prompt, context)
            
            # Cache result
            result = (answer, docs)
            await self.cache_service.cache_response(
                question, self.config.llm_model, k, result
            )
            
            logger.info(f"Query processed successfully, retrieved {len(docs)} documents")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise


# Convenience function
def create_hr_service(container: Optional[ServiceContainer] = None) -> HRChatbotService:
    """Create HR chatbot service with dependency injection."""
    if container is None:
        container = ServiceFactory.create_default_container()
    return HRChatbotService(container)