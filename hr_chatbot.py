"""
hr_chatbot.py - Core functions for HR Policy Chatbot.

Dependencies:
    pip install langchain openai faiss-cpu tiktoken

Ensure OPENAI_API_KEY is set in your environment.
"""
import os
import logging
from typing import List, Tuple, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from chunking import create_chunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config import get_config
from cache import get_query_cache, get_embedding_cache
from validation import get_validator, ValidationError

# Get configuration, cache, and validator instances
config = get_config()
query_cache = get_query_cache()
embedding_cache = get_embedding_cache()
validator = get_validator()

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format=config.log_format
)
logger = logging.getLogger(__name__)

def load_and_split_pdfs(
    folder_path: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List:
    """
    Load all PDF files from a folder and split into token-based chunks.
    Returns a list of LangChain Document objects, each with metadata.source set.
    
    Args:
        folder_path: Path to folder containing PDF files
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of Document objects with content and metadata
        
    Raises:
        FileNotFoundError: If folder_path doesn't exist
        ValueError: If no PDF files found in folder
    """
    # Use config defaults if not provided
    folder_path = folder_path or config.policies_folder
    chunk_size = chunk_size or config.chunk_size
    chunk_overlap = chunk_overlap or config.chunk_overlap
    
    try:
        # Validate inputs
        validated_folder = validator.validate_folder_path(folder_path)
        chunk_size, chunk_overlap = validator.validate_chunk_params(chunk_size, chunk_overlap)
        folder_path = str(validated_folder)
            
        # Collect PDF paths
        # Get PDF files and validate each one
        pdf_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(config.allowed_file_extensions)
        ]
        
        pdf_paths = []
        for f in pdf_files:
            file_path = os.path.join(folder_path, f)
            try:
                validator.validate_pdf_file(file_path)
                pdf_paths.append(file_path)
            except ValidationError as e:
                logger.warning(f"Skipping invalid file {f}: {str(e)}")
                continue
        
        if not pdf_paths:
            logger.warning(f"No PDF files found in {folder_path}")
            raise ValueError(f"No PDF files found in {folder_path}")
            
        logger.info(f"Found {len(pdf_paths)} PDF files to process")
        
    except PermissionError:
        logger.error(f"Permission denied accessing folder: {folder_path}")
        raise
    # Initialize smart chunker
    try:
        chunker = create_chunker(
            strategy="smart",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        documents = []
        
        for path in pdf_paths:
            try:
                logger.info(f"Processing PDF: {os.path.basename(path)}")
                loader = PyPDFLoader(path)
                docs = loader.load()
                
                if not docs:
                    logger.warning(f"No content extracted from {path}")
                    continue
                    
                # Tag source in metadata
                for doc in docs:
                    doc.metadata["source"] = path
                
                # Use smart chunking for better quality
                splits = chunker.chunk_documents(docs)
                documents.extend(splits)
                logger.info(f"Created {len(splits)} chunks from {os.path.basename(path)}")
                
            except Exception as e:
                logger.error(f"Error processing {path}: {str(e)}")
                continue
                
        logger.info(f"Total documents processed: {len(documents)}")
        return documents
        
    except Exception as e:
        logger.error(f"Error in document processing: {str(e)}")
        raise

def build_vectorstore(
    documents,
    embedding_model_name: Optional[str] = None,
    index_path: Optional[str] = None,
) -> Optional[FAISS]:
    """
    Build a FAISS vector store from documents and save locally.
    
    Args:
        documents: List of Document objects to embed
        embedding_model_name: OpenAI embedding model name
        index_path: Path to save the FAISS index
        
    Returns:
        FAISS vectorstore instance or None if failed
        
    Raises:
        ValueError: If documents list is empty
        Exception: If embedding or index creation fails
    """
    # Use config defaults if not provided
    embedding_model_name = embedding_model_name or config.embedding_model
    index_path = index_path or config.index_path
    
    try:
        if not documents:
            logger.error("No documents provided for vectorstore creation")
            raise ValueError("Documents list cannot be empty")
            
        logger.info(f"Creating embeddings for {len(documents)} documents")
        embeddings = OpenAIEmbeddings(
            model=embedding_model_name,
            openai_api_key=config.openai_api_key
        )
        
        logger.info("Building FAISS vectorstore")
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        logger.info(f"Saving vectorstore to {index_path}")
        vectorstore.save_local(index_path)
        
        logger.info("Vectorstore created and saved successfully")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error building vectorstore: {str(e)}")
        raise

def load_vectorstore(
    index_path: Optional[str] = None,
    embedding_model_name: Optional[str] = None,
) -> Optional[FAISS]:
    """
    Load a FAISS vector store from local disk.
    
    Args:
        index_path: Path to the saved FAISS index
        embedding_model_name: OpenAI embedding model name
        
    Returns:
        FAISS vectorstore instance or None if failed
        
    Raises:
        FileNotFoundError: If index path doesn't exist
        Exception: If loading fails
    """
    # Use config defaults if not provided
    index_path = index_path or config.index_path
    embedding_model_name = embedding_model_name or config.embedding_model
    
    try:
        if not os.path.exists(index_path):
            logger.error(f"Index path does not exist: {index_path}")
            raise FileNotFoundError(f"Index path does not exist: {index_path}")
            
        logger.info(f"Loading vectorstore from {index_path}")
        embeddings = OpenAIEmbeddings(
            model=embedding_model_name,
            openai_api_key=config.openai_api_key
        )
        
        # Allow deserializing our own index (pickle) since we trust our local files
        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        
        logger.info("Vectorstore loaded successfully")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error loading vectorstore: {str(e)}")
        raise

def answer_query(
    question: str,
    vectorstore: FAISS,
    llm_model_name: Optional[str] = None,
    k: Optional[int] = None,
) -> Tuple[str, List]:
    """
    Use RetrievalQA to answer a question given a vectorstore.
    
    Args:
        question: User's question
        vectorstore: FAISS vectorstore instance
        llm_model_name: OpenAI model name for generation
        k: Number of documents to retrieve
        
    Returns:
        Tuple of (answer_string, source_documents)
        
    Raises:
        ValueError: If question is empty or vectorstore is None
        Exception: If query processing fails
    """
    # Use config defaults if not provided
    llm_model_name = llm_model_name or config.llm_model
    k = k or config.default_k
    
    try:
        # Validate inputs
        question = validator.validate_query(question)
        llm_model_name = validator.validate_model_name(llm_model_name)
        k = validator.validate_k_value(k)
            
        if vectorstore is None:
            logger.error("Vectorstore is None")
            raise ValueError("Vectorstore cannot be None")
            
        logger.info(f"Processing query: {question[:100]}...")
        
        # Check cache first
        cached_result = query_cache.get(question, llm_model_name, k)
        if cached_result is not None:
            return cached_result
        
        llm = ChatOpenAI(
            model=llm_model_name, 
            temperature=0,
            openai_api_key=config.openai_api_key
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        
        # Prompt: ground answers strictly in the given context
        prompt = PromptTemplate(
            template=(
                "You are a helpful HR assistant. Below are excerpts from the company's policies.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n"
                "Answer concisely based ONLY on the provided context.\n"
                "If the information is not present, respond exactly with \"I don't know.\""
            ),
            input_variables=["context", "question"],
        )
        
        # Build a RetrievalQA chain using the customized prompt
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        
        # Use invoke() instead of deprecated direct call
        result = qa.invoke(question)
        answer = result["result"].strip()
        docs = result.get("source_documents", [])
        
        logger.info(f"Query processed successfully, retrieved {len(docs)} documents")
        
        # Cache the result
        result = (answer, docs)
        query_cache.set(question, llm_model_name, k, result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise