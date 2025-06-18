"""
app_launcher.py - Application launcher for Document Chatbot Framework.

This script provides a convenient way to launch different versions of the chatbot:
1. Domain-based chatbot (original)
2. Interactive upload chatbot (new)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def launch_domain_app(domain: str = None):
    """Launch the domain-based chatbot application."""
    print("🚀 Launching Domain-Based Chatbot...")
    
    env = os.environ.copy()
    if domain:
        env['CHATBOT_DOMAIN'] = domain
        print(f"📋 Using domain: {domain}")
    
    # Launch streamlit app
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py"
    ], env=env)

def launch_interactive_app():
    """Launch the interactive upload chatbot application."""
    print("📚 Launching Interactive Upload Chatbot...")
    
    # Launch interactive streamlit app
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "streamlit_interactive.py"
    ])

def list_available_domains():
    """List available domains."""
    domains_dir = Path("domains")
    if not domains_dir.exists():
        print("❌ No domains directory found")
        return []
    
    domains = []
    for yaml_file in domains_dir.glob("*.yaml"):
        domain_name = yaml_file.stem
        domains.append(domain_name)
    
    return domains

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(
        description="Document Chatbot Framework Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app_launcher.py --interactive          # Launch interactive upload app
  python app_launcher.py --domain hr            # Launch HR domain app
  python app_launcher.py --domain legal         # Launch legal domain app
  python app_launcher.py --list-domains         # List available domains
        """
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive upload chatbot"
    )
    
    parser.add_argument(
        "--domain",
        type=str,
        help="Launch domain-based chatbot with specified domain"
    )
    
    parser.add_argument(
        "--list-domains",
        action="store_true",
        help="List available domains"
    )
    
    args = parser.parse_args()
    
    # List domains
    if args.list_domains:
        domains = list_available_domains()
        if domains:
            print("📋 Available domains:")
            for domain in sorted(domains):
                print(f"  • {domain}")
        else:
            print("❌ No domains found in domains/ directory")
        return
    
    # Launch interactive app
    if args.interactive:
        launch_interactive_app()
        return
    
    # Launch domain app
    if args.domain:
        domains = list_available_domains()
        if args.domain not in domains:
            print(f"❌ Domain '{args.domain}' not found")
            print(f"Available domains: {', '.join(domains)}")
            return
        
        launch_domain_app(args.domain)
        return
    
    # Interactive selection if no arguments provided
    print("🤖 Document Chatbot Framework Launcher")
    print("=" * 50)
    print()
    print("Choose an option:")
    print("1. 📚 Interactive Upload Chatbot (upload your own documents)")
    print("2. 📋 Domain-Based Chatbot (pre-configured domains)")
    print("3. 📝 List Available Domains")
    print("4. ❌ Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == "1":
                launch_interactive_app()
                break
            
            elif choice == "2":
                domains = list_available_domains()
                if not domains:
                    print("❌ No domains found. Create domains first or use interactive mode.")
                    continue
                
                print("\nAvailable domains:")
                for i, domain in enumerate(sorted(domains), 1):
                    print(f"  {i}. {domain}")
                
                try:
                    domain_choice = input(f"\nChoose domain (1-{len(domains)}): ").strip()
                    domain_index = int(domain_choice) - 1
                    
                    if 0 <= domain_index < len(domains):
                        selected_domain = sorted(domains)[domain_index]
                        launch_domain_app(selected_domain)
                        break
                    else:
                        print("❌ Invalid choice. Try again.")
                        
                except ValueError:
                    print("❌ Please enter a valid number.")
                    continue
            
            elif choice == "3":
                domains = list_available_domains()
                if domains:
                    print("\n📋 Available domains:")
                    for domain in sorted(domains):
                        print(f"  • {domain}")
                else:
                    print("\n❌ No domains found in domains/ directory")
                print()
                continue
            
            elif choice == "4":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                continue
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            continue

if __name__ == "__main__":
    main()