"""
Advanced document chunking strategies for HR Chatbot.

This module provides optimized chunking strategies to improve
retrieval accuracy and quality.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_text_splitters import (
    TokenTextSplitter, 
    RecursiveCharacterTextSplitter,
    NLTKTextSplitter
)
from langchain_core.documents import Document
from config import get_config

config = get_config()
logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies."""
    chunk_size: int = 500
    chunk_overlap: int = 100
    strategy: str = "smart"  # "token", "recursive", "semantic", "smart"
    preserve_structure: bool = True
    min_chunk_size: int = 50
    max_chunk_size: int = 2000


class SmartChunker:
    """
    Advanced chunker that uses multiple strategies based on content type.
    """
    
    def __init__(self, config: ChunkingConfig = None):
        """
        Initialize the smart chunker.
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig()
        
        # Initialize different splitters
        self.token_splitter = TokenTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        # Pattern for detecting sections/headers
        self.section_pattern = re.compile(
            r'^(\s*(?:\d+\.?|\w+\.|\*|\-)\s*[A-Z][^\n]*?)$',
            re.MULTILINE
        )
        
        # Pattern for detecting policy-specific structures
        self.policy_patterns = {
            'procedure': re.compile(r'(?i)(procedure|process|steps?|method)', re.MULTILINE),
            'requirement': re.compile(r'(?i)(must|shall|required?|mandatory)', re.MULTILINE),
            'definition': re.compile(r'(?i)(means|defined? as|refers? to)', re.MULTILINE),
            'example': re.compile(r'(?i)(example|for instance|such as)', re.MULTILINE),
        }
    
    def _detect_content_type(self, text: str) -> str:
        """
        Detect the type of content to choose appropriate chunking strategy.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Content type identifier
        """
        text_lower = text.lower()
        
        # Check for structured policy content
        if any(pattern.search(text) for pattern in self.policy_patterns.values()):
            return "policy"
        
        # Check for lists/procedures
        if re.search(r'^\s*\d+\.', text, re.MULTILINE) or \
           re.search(r'^\s*[a-z]\)', text, re.MULTILINE):
            return "procedure"
        
        # Check for definitions/glossary
        if text_lower.count('means') > 2 or text_lower.count('definition') > 1:
            return "definition"
        
        # Check for tables/structured data
        if text.count('|') > 5 or text.count('\t') > 10:
            return "tabular"
        
        # Default to general content
        return "general"
    
    def _preserve_sentence_boundaries(self, chunks: List[str]) -> List[str]:
        """
        Ensure chunks don't break in the middle of sentences.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of adjusted chunks
        """
        adjusted_chunks = []
        
        for chunk in chunks:
            # Check if chunk ends mid-sentence
            if chunk and not re.search(r'[.!?]\s*$', chunk.strip()):
                # Find the last complete sentence
                sentences = re.split(r'([.!?]\s+)', chunk)
                if len(sentences) > 2:
                    # Keep complete sentences, move incomplete to next chunk
                    complete_part = ''.join(sentences[:-1]).strip()
                    if complete_part:
                        adjusted_chunks.append(complete_part)
                    # Note: In a real implementation, we'd need to handle
                    # the incomplete part by adding it to the next chunk
                else:
                    adjusted_chunks.append(chunk)
            else:
                adjusted_chunks.append(chunk)
        
        return [chunk for chunk in adjusted_chunks if chunk.strip()]
    
    def _chunk_policy_content(self, text: str) -> List[str]:
        """
        Specialized chunking for policy documents.
        
        Args:
            text: Policy text content
            
        Returns:
            List of chunks optimized for policy content
        """
        chunks = []
        
        # Split by major sections first
        sections = re.split(r'\n\s*(?=\d+\.?\s+[A-Z])', text)
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # If section is small enough, keep as one chunk
            if len(section) <= self.config.chunk_size:
                chunks.append(section)
            else:
                # Split large sections by subsections or paragraphs
                subsections = re.split(r'\n\s*(?=\w+\.?\s)', section)
                
                current_chunk = ""
                for subsection in subsections:
                    if len(current_chunk + subsection) <= self.config.chunk_size:
                        current_chunk += "\n" + subsection if current_chunk else subsection
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = subsection
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_procedure_content(self, text: str) -> List[str]:
        """
        Specialized chunking for procedural content.
        
        Args:
            text: Procedural text content
            
        Returns:
            List of chunks optimized for procedures
        """
        # Split by numbered steps first
        steps = re.split(r'\n\s*(?=\d+\.)', text)
        
        chunks = []
        current_chunk = ""
        
        for step in steps:
            step = step.strip()
            if not step:
                continue
            
            # Try to keep related steps together
            if len(current_chunk + step) <= self.config.chunk_size:
                current_chunk += "\n" + step if current_chunk else step
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If single step is too long, split it
                if len(step) > self.config.chunk_size:
                    step_chunks = self.recursive_splitter.split_text(step)
                    chunks.extend(step_chunks)
                    current_chunk = ""
                else:
                    current_chunk = step
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_document(self, document: Document) -> List[Document]:
        """
        Chunk a document using the smart strategy.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of chunked documents
        """
        text = document.page_content
        content_type = self._detect_content_type(text)
        
        logger.debug(f"Detected content type: {content_type}")
        
        # Choose chunking strategy based on content type
        if content_type == "policy":
            chunks = self._chunk_policy_content(text)
        elif content_type == "procedure":
            chunks = self._chunk_procedure_content(text)
        else:
            # Use recursive splitter for general content
            chunks = self.recursive_splitter.split_text(text)
        
        # Preserve sentence boundaries if configured
        if self.config.preserve_structure:
            chunks = self._preserve_sentence_boundaries(chunks)
        
        # Filter chunks by size
        filtered_chunks = [
            chunk for chunk in chunks
            if self.config.min_chunk_size <= len(chunk) <= self.config.max_chunk_size
        ]
        
        # Create Document objects with metadata
        chunked_docs = []
        for i, chunk in enumerate(filtered_chunks):
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                'chunk_id': i,
                'chunk_type': content_type,
                'total_chunks': len(filtered_chunks),
                'chunk_size': len(chunk)
            })
            
            chunked_docs.append(Document(
                page_content=chunk,
                metadata=chunk_metadata
            ))
        
        logger.info(f"Split document into {len(chunked_docs)} chunks using {content_type} strategy")
        return chunked_docs
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of all chunked documents
        """
        all_chunks = []
        
        for doc in documents:
            try:
                chunks = self.chunk_document(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error chunking document {doc.metadata.get('source', 'unknown')}: {str(e)}")
                # Fallback to simple chunking
                try:
                    fallback_chunks = self.token_splitter.split_documents([doc])
                    all_chunks.extend(fallback_chunks)
                except Exception as e2:
                    logger.error(f"Fallback chunking also failed: {str(e2)}")
                    continue
        
        return all_chunks


class SemanticChunker:
    """
    Experimental semantic chunker that groups related content.
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize semantic chunker.
        
        Args:
            similarity_threshold: Threshold for semantic similarity
        """
        self.similarity_threshold = similarity_threshold
        self.base_chunker = SmartChunker()
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents using semantic similarity.
        
        Note: This is a placeholder for semantic chunking.
        Real implementation would require embedding-based similarity.
        
        Args:
            documents: Documents to chunk
            
        Returns:
            Semantically chunked documents
        """
        # For now, fall back to smart chunking
        # TODO: Implement embedding-based semantic chunking
        logger.info("Semantic chunking not fully implemented, using smart chunking")
        return self.base_chunker.chunk_documents(documents)


def create_chunker(strategy: str = None, **kwargs) -> SmartChunker:
    """
    Factory function to create a chunker.
    
    Args:
        strategy: Chunking strategy to use
        **kwargs: Additional configuration options
        
    Returns:
        Configured chunker instance
    """
    strategy = strategy or config.chunk_size
    
    chunking_config = ChunkingConfig(
        chunk_size=kwargs.get('chunk_size', config.chunk_size),
        chunk_overlap=kwargs.get('chunk_overlap', config.chunk_overlap),
        strategy=strategy,
        preserve_structure=kwargs.get('preserve_structure', True),
        min_chunk_size=kwargs.get('min_chunk_size', 50),
        max_chunk_size=kwargs.get('max_chunk_size', 2000)
    )
    
    if strategy == "semantic":
        return SemanticChunker()
    else:
        return SmartChunker(chunking_config)