import argparse
import os
import logging
from document_chatbot import DocumentChatbot
from domain_config import load_current_domain_config, get_available_domains
from config import get_config

# Get configuration
config = get_config()
logger = logging.getLogger(__name__)

def main():
    # Load domain configuration
    try:
        domain_config = load_current_domain_config()
        chatbot = DocumentChatbot(domain_config)
    except Exception as e:
        print(f"❌ Error loading domain configuration: {str(e)}")
        print(f"Available domains: {', '.join(get_available_domains())}")
        print(f"Set CHATBOT_DOMAIN environment variable to one of the available domains.")
        return
    
    parser = argparse.ArgumentParser(
        description=f"{domain_config.name} CLI",
        epilog=f"Domain: {domain_config.domain} | {domain_config.description}"
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help=f"Build FAISS index from {domain_config.documents_folder_display_name.lower()}",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=domain_config.documents_folder,
        help=f"Folder containing {domain_config.documents_folder_display_name.lower()}",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default=domain_config.documents_index_path,
        help="Path to save/load FAISS index",
    )
    parser.add_argument(
        "--query",
        type=str,
        help=f"Question to ask the {domain_config.name.lower()}",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=domain_config.default_k,
        help="Number of source documents to retrieve",
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="Override domain (available: " + ", ".join(get_available_domains()) + ")",
    )
    args = parser.parse_args()
    
    # Handle domain override
    if args.domain:
        os.environ['CHATBOT_DOMAIN'] = args.domain
        try:
            domain_config = load_current_domain_config()
            chatbot = DocumentChatbot(domain_config)
            print(f"Switched to domain: {domain_config.domain}")
        except Exception as e:
            print(f"❌ Error switching to domain '{args.domain}': {str(e)}")
            return

    if args.index:
        try:
            logger.info(f"Loading and splitting documents from '{args.folder}'...")
            print(f"Loading and splitting {domain_config.documents_folder_display_name.lower()} from '{args.folder}'...")
            docs = chatbot.load_and_split_documents(args.folder)
            
            logger.info(f"Building FAISS index and saving to '{args.index_path}'...")
            print(f"Building FAISS index and saving to '{args.index_path}'...")
            chatbot.build_vectorstore(docs, index_path=args.index_path)
            
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
            vectorstore = chatbot.load_vectorstore(args.index_path)
            
            logger.info(f"Processing query: {args.query}")
            print(f"Asking {domain_config.name.lower()}: {args.query}")
            answer, docs = chatbot.answer_query(args.query, vectorstore, k=args.k)
            
            print("\nAnswer:\n", answer)
            print(f"\nSource {domain_config.documents_folder_display_name}:")
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