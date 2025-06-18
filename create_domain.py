#!/usr/bin/env python3
"""
Domain Template Generator for Document Chatbot Framework

This script helps create new domain configurations for different types
of document chatbots (HR, Legal, Technical, etc.).

Usage:
    python create_domain.py --domain mydomain --name "My Domain Chatbot"
    python create_domain.py --interactive
"""

import argparse
import os
import yaml
from pathlib import Path
from typing import Dict, Any


def get_domain_template() -> Dict[str, Any]:
    """Get the base domain configuration template."""
    return {
        'name': '',
        'description': '',
        'domain': '',
        'ui': {
            'title': '',
            'page_title': '',
            'page_icon': '🤖',
            'sidebar_title': '⚙️ Configuration',
            'footer': 'Built with Streamlit, LangChain, and OpenAI'
        },
        'documents': {
            'folder': '',
            'folder_display_name': '',
            'file_types': ['.pdf'],
            'index_path': ''
        },
        'query': {
            'placeholder': '',
            'help_text': '',
            'button_text': '🔍 Get Answer'
        },
        'prompts': {
            'system_prompt': '''You are a helpful assistant. Below are excerpts from documents.

Context:
{context}

Question: {question}
Answer concisely based ONLY on the provided context.
If the information is not present, respond exactly with "I don't know."'''
        },
        'processing': {
            'chunk_size': 500,
            'chunk_overlap': 100,
            'default_k': 4,
            'max_k': 10
        },
        'messages': {
            'no_folder': '📁 {folder} folder not found!',
            'no_folder_help': 'Please ensure your documents are in the {folder} folder.',
            'no_api_key': '⚠️ OpenAI API key not found! Please add it to Streamlit secrets.',
            'api_key_help': 'For Streamlit Cloud: Add OPENAI_API_KEY in the app\'s secrets section.',
            'no_question': '⚠️ Please enter a question.',
            'processing_error': '❌ Error processing query: {error}',
            'index_ready': '✅ Search index ready!',
            'found_files': '📁 Found {count} documents'
        }
    }


def create_domain_interactive():
    """Create a domain configuration interactively."""
    print("🚀 Document Chatbot Domain Generator")
    print("=" * 50)
    
    # Get basic information
    domain = input("Domain ID (lowercase, no spaces): ").strip().lower()
    if not domain:
        print("❌ Domain ID is required!")
        return
    
    name = input(f"Display name (e.g., 'Legal Document Assistant'): ").strip()
    if not name:
        name = f"{domain.title()} Assistant"
    
    description = input("Description: ").strip()
    if not description:
        description = f"Search and query {domain} documents"
    
    # Get UI configuration
    print("\n📱 UI Configuration")
    page_icon = input("Page icon (emoji, default 🤖): ").strip() or "🤖"
    
    # Get document configuration
    print("\n📁 Document Configuration")
    folder = input(f"Documents folder (default ./{domain}_docs): ").strip()
    if not folder:
        folder = f"./{domain}_docs"
    
    folder_display_name = input(f"Folder display name (default '{domain.title()} Documents'): ").strip()
    if not folder_display_name:
        folder_display_name = f"{domain.title()} Documents"
    
    # Get file types
    file_types_input = input("Supported file types (comma-separated, default .pdf): ").strip()
    if file_types_input:
        file_types = [f".{ext.strip().lstrip('.')}" for ext in file_types_input.split(",")]
    else:
        file_types = [".pdf"]
    
    # Get query configuration
    print("\n💬 Query Configuration")
    placeholder = input(f"Query placeholder (default 'e.g., What is the {domain} policy?'): ").strip()
    if not placeholder:
        placeholder = f"e.g., What is the {domain} policy?"
    
    help_text = input(f"Help text (default 'Ask questions about {domain} documents'): ").strip()
    if not help_text:
        help_text = f"Ask questions about {domain} documents"
    
    # Build configuration
    config = get_domain_template()
    config['name'] = name
    config['description'] = description
    config['domain'] = domain
    
    # UI config
    config['ui']['title'] = f"{page_icon} {name}"
    config['ui']['page_title'] = name
    config['ui']['page_icon'] = page_icon
    
    # Documents config
    config['documents']['folder'] = folder
    config['documents']['folder_display_name'] = folder_display_name
    config['documents']['file_types'] = file_types
    config['documents']['index_path'] = f"faiss_index_{domain}"
    
    # Query config
    config['query']['placeholder'] = placeholder
    config['query']['help_text'] = help_text
    config['query']['button_text'] = f"🔍 Search {folder_display_name}"
    
    return config, domain


def create_domain_from_args(domain: str, name: str, description: str = None, icon: str = "🤖"):
    """Create domain configuration from command line arguments."""
    if not domain or not name:
        raise ValueError("Domain and name are required")
    
    description = description or f"Search and query {domain} documents"
    
    config = get_domain_template()
    config['name'] = name
    config['description'] = description
    config['domain'] = domain
    
    # UI config
    config['ui']['title'] = f"{icon} {name}"
    config['ui']['page_title'] = name
    config['ui']['page_icon'] = icon
    
    # Documents config
    config['documents']['folder'] = f"./{domain}_docs"
    config['documents']['folder_display_name'] = f"{domain.title()} Documents"
    config['documents']['index_path'] = f"faiss_index_{domain}"
    
    # Query config
    config['query']['placeholder'] = f"e.g., What is the {domain} policy?"
    config['query']['help_text'] = f"Ask questions about {domain} documents"
    config['query']['button_text'] = f"🔍 Search {domain.title()} Docs"
    
    return config, domain


def save_domain_config(config: Dict[str, Any], domain: str, domains_folder: str = "./domains"):
    """Save domain configuration to YAML file."""
    domains_path = Path(domains_folder)
    domains_path.mkdir(exist_ok=True)
    
    config_file = domains_path / f"{domain}.yaml"
    
    # Check if file already exists
    if config_file.exists():
        overwrite = input(f"⚠️  Domain '{domain}' already exists. Overwrite? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("❌ Cancelled.")
            return False
    
    # Add header comment
    header = f"# {config['name']} Domain Configuration\n"
    
    # Save to YAML file
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
    
    print(f"✅ Domain configuration saved to: {config_file}")
    return True


def create_domain_folder(domain: str, folder_path: str):
    """Create the documents folder for the domain."""
    folder = Path(folder_path)
    folder.mkdir(exist_ok=True)
    
    # Create README file
    readme_content = f"""# {domain.title()} Documents Folder

Place your {domain} documents in this folder.

**Supported file types:**
- PDF files (.pdf)

**Examples of documents to include:**
- {domain.title()} policies
- {domain.title()} procedures
- {domain.title()} documentation
- Related compliance materials

The {domain} chatbot will process all supported files in this folder and make them searchable through natural language queries.
"""
    
    readme_file = folder / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✅ Documents folder created: {folder}")
    print(f"📄 README created: {readme_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a new domain configuration for the Document Chatbot Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_domain.py --interactive
  python create_domain.py --domain medical --name "Medical Records Assistant"
  python create_domain.py --domain legal --name "Legal Document Assistant" --icon "⚖️"
        """
    )
    
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="Domain ID (lowercase, no spaces)"
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Display name for the chatbot"
    )
    parser.add_argument(
        "--description",
        type=str,
        help="Description of the chatbot"
    )
    parser.add_argument(
        "--icon",
        type=str,
        default="🤖",
        help="Page icon (emoji, default: 🤖)"
    )
    parser.add_argument(
        "--domains-folder",
        type=str,
        default="./domains",
        help="Folder to save domain configurations (default: ./domains)"
    )
    parser.add_argument(
        "--create-folder",
        action="store_true",
        help="Create the documents folder for the domain"
    )
    
    args = parser.parse_args()
    
    try:
        if args.interactive:
            config, domain = create_domain_interactive()
        elif args.domain and args.name:
            config, domain = create_domain_from_args(
                args.domain, args.name, args.description, args.icon
            )
        else:
            parser.print_help()
            return
        
        # Save configuration
        if save_domain_config(config, domain, args.domains_folder):
            print(f"\n🎉 Domain '{domain}' created successfully!")
            
            # Create folder if requested
            if args.create_folder:
                create_domain_folder(domain, config['documents']['folder'])
            
            print(f"\n📖 Usage:")
            print(f"  CHATBOT_DOMAIN={domain} python cli.py --index")
            print(f"  CHATBOT_DOMAIN={domain} python cli.py --query 'Your question here'")
            print(f"  CHATBOT_DOMAIN={domain} streamlit run streamlit_app.py")
            
            if not args.create_folder:
                print(f"\n💡 Tip: Run with --create-folder to automatically create the documents folder")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())