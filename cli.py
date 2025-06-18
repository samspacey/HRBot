import argparse
import os
import logging
from hr_chatbot import (
    load_and_split_pdfs,
    build_vectorstore,
    load_vectorstore,
    answer_query,
)
from config import get_config

# Get configuration
config = get_config()
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="HR Policy Chatbot CLI")
    parser.add_argument(
        "--index",
        action="store_true",
        help="Build FAISS index from PDF policies",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=config.policies_folder,
        help="Folder containing PDF policy documents",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default=config.index_path,
        help="Path to save/load FAISS index",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Question to ask the chatbot",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=config.default_k,
        help="Number of source documents to retrieve",
    )
    args = parser.parse_args()

    if args.index:
        try:
            logger.info(f"Loading and splitting PDFs from '{args.folder}'...")
            print(f"Loading and splitting PDFs from '{args.folder}'...")
            docs = load_and_split_pdfs(args.folder)
            
            logger.info(f"Building FAISS index and saving to '{args.index_path}'...")
            print(f"Building FAISS index and saving to '{args.index_path}'...")
            build_vectorstore(docs, index_path=args.index_path)
            
            logger.info("Index built successfully")
            print("Index built successfully.")
        except Exception as e:
            logger.error(f"Error building index: {str(e)}")
            print(f"Error building index: {str(e)}")
            return

    if args.query:
        try:
            if not os.path.exists(args.index_path):
                logger.error(f"Index path '{args.index_path}' does not exist")
                print(
                    f"Index path '{args.index_path}' does not exist. Please run with --index first."
                )
                return
                
            logger.info(f"Loading vectorstore from {args.index_path}")
            vectorstore = load_vectorstore(args.index_path)
            
            logger.info(f"Processing query: {args.query}")
            print(f"Asking question: {args.query}")
            answer, docs = answer_query(args.query, vectorstore, k=args.k)
            
            print("\nAnswer:\n", answer)
            print("\nSource Documents:")
            for i, doc in enumerate(docs, start=1):
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "unknown")
                print(f"{i}. {source} (page {page})")
                print(doc.page_content[:500])
                print("-----")
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            print(f"Error processing query: {str(e)}")
            return

if __name__ == "__main__":
    main()