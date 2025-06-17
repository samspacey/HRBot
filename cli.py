import argparse
import os
from hr_chatbot import (
    load_and_split_pdfs,
    build_vectorstore,
    load_vectorstore,
    answer_query,
)

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
        default="./policies",
        help="Folder containing PDF policy documents",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="faiss_index_hr",
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
        default=4,
        help="Number of source documents to retrieve",
    )
    args = parser.parse_args()

    if args.index:
        print(f"Loading and splitting PDFs from '{args.folder}'...")
        docs = load_and_split_pdfs(args.folder)
        print(f"Building FAISS index and saving to '{args.index_path}'...")
        build_vectorstore(docs, index_path=args.index_path)
        print("Index built successfully.")

    if args.query:
        if not os.path.exists(args.index_path):
            print(
                f"Index path '{args.index_path}' does not exist. Please run with --index first."
            )
            return
        vectorstore = load_vectorstore(args.index_path)
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

if __name__ == "__main__":
    main()