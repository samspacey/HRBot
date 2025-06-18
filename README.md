# Document Chatbot Framework

An enterprise-grade intelligent chatbot framework that enables organizations to create domain-specific document query assistants using natural language. The system uses Retrieval-Augmented Generation (RAG) architecture with advanced document processing, caching, and validation to provide accurate, context-aware answers from any document collection.

**🎯 Multi-Domain Support:** Easily create chatbots for HR policies, legal documents, technical documentation, financial procedures, and more with simple YAML configuration files.

## ✨ Features

- **🎯 Multi-Domain Framework**: Create chatbots for any document type (HR, Legal, Technical, Financial, etc.)
- **⚙️ YAML Configuration**: Simple configuration files for different domains and use cases
- **🔍 Advanced Document Processing**: Smart chunking strategies for optimal retrieval accuracy
- **⚡ High Performance**: FAISS vector store with caching for sub-second response times
- **🛡️ Enterprise Security**: Input validation, sanitization, and security scanning
- **🚀 Multiple Interfaces**: CLI, Streamlit web app, Python API, and async support
- **📊 Comprehensive Monitoring**: Logging, metrics, and performance tracking
- **🐳 Production Ready**: Docker support, pre-commit hooks, and CI/CD integration
- **🧪 Robust Testing**: Unit, integration, and performance tests with >90% coverage

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Docker (optional)

### 🎯 **How to Run the App** (4 Simple Steps)

1. **📦 Setup Environment:**
```bash
git clone <repository-url>
cd HRBot
make setup-dev  # Installs dependencies and creates .env file
```

2. **🔑 Add Your OpenAI API Key:**
```bash
# Edit the .env file that was created:
nano .env  # or use any text editor
# Add: OPENAI_API_KEY=your_actual_openai_api_key_here
```

3. **📚 Build Search Index:**
```bash
make index  # Processes PDF files in policies/ folder (HR domain by default)
```

4. **🚀 Start the App:**
```bash
make serve  # Opens web interface at http://localhost:8501
```

**That's it!** 🎉 Your document chatbot is now running!

### 🎯 **Creating Different Domain Chatbots**

Want to create a chatbot for legal documents, technical docs, or financial policies? Use the domain generator:

```bash
# Interactive mode - guided setup
python create_domain.py --interactive

# Command line mode - quick setup
python create_domain.py --domain legal --name "Legal Document Assistant" --icon "⚖️"

# Switch to your new domain
CHATBOT_DOMAIN=legal python cli.py --index
CHATBOT_DOMAIN=legal streamlit run streamlit_app.py
```

### Alternative Ways to Run

**Command Line Interface:**
```bash
make query QUESTION="What is the vacation policy?"
# Or for different domains:
CHATBOT_DOMAIN=legal make query QUESTION="What are the contract terms?"
```

**Docker (if you prefer containers):**
```bash
docker build -t document-chatbot .
docker run -p 8501:8501 --env-file .env document-chatbot
```

**Development Mode (with hot reload):**
```bash
make docker-dev  # Full development environment
```

### ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| "No index found" | Run `make index` first |
| API key errors | Check your `.env` file has `OPENAI_API_KEY=your_key` |
| Dependencies fail | Run `pip install -r requirements.txt` |
| Port already in use | Kill other Streamlit processes or use different port |

**Need help?** Run `make help` to see all available commands.

## 📚 Usage

### Development Commands

The Makefile provides convenient commands for all development tasks:

```bash
make help              # Show all available commands
make setup-dev         # Set up development environment
make test              # Run all tests
make lint              # Run code linting
make format            # Format code with black and isort
make serve             # Start web interface
make index             # Build search index
make docker-dev        # Start development with Docker
```

### Web Interface (Streamlit)

```bash
make serve
# Or directly: streamlit run streamlit_app.py
```

Navigate to `http://localhost:8501` for the interactive chatbot interface.

### Command Line Interface

```bash
# Build the index
make index

# Ask questions
make query QUESTION="What is the sick leave policy?"

# Direct CLI usage
python cli.py --query "How many vacation days do I get?" --k 5
```

### Python API

```python
from hr_chatbot import load_vectorstore, answer_query
from services import create_hr_service

# Method 1: Direct usage
vectorstore = load_vectorstore("faiss_index_hr")
answer, sources = answer_query("What is the remote work policy?", vectorstore)

# Method 2: Service-based approach (recommended)
hr_service = create_hr_service()
vectorstore = await hr_service.load_index()
answer, sources = await hr_service.answer_query("What is the remote work policy?", vectorstore)
```

### Async Usage

```python
import asyncio
from async_hr_chatbot import AsyncHRChatbot

async def main():
    async with AsyncHRChatbot() as chatbot:
        vectorstore = await chatbot.load_vectorstore_async()
        answer, docs = await chatbot.answer_query_async(
            "What is the vacation policy?", 
            vectorstore
        )
        print(answer)

asyncio.run(main())
```

## 🎯 Domain Configuration

The framework supports multiple document types through YAML configuration files. Each domain can have its own branding, prompts, and document processing settings.

### Available Domains

- **HR** (`domains/hr.yaml`) - Human resources policies and procedures
- **Legal** (`domains/legal.yaml`) - Legal documents and contracts
- **Technical** (`domains/technical.yaml`) - Technical documentation and manuals
- **Financial** (`domains/financial.yaml`) - Financial policies and procedures

### Creating New Domains

Use the domain generator to create new chatbot configurations:

```bash
# Interactive mode with guided setup
python create_domain.py --interactive

# Command line mode for quick setup
python create_domain.py --domain medical --name "Medical Records Assistant" --icon "🏥"

# Create with custom settings
python create_domain.py --domain compliance \
  --name "Compliance Assistant" \
  --description "Search compliance documents and regulations" \
  --icon "📋" \
  --create-folder
```

### Domain Configuration Structure

Each domain configuration includes:

```yaml
name: "Legal Document Assistant"
description: "Search and query legal documents"
domain: "legal"
ui:
  title: "⚖️ Legal Document Assistant"
  page_icon: "⚖️"
documents:
  folder: "./legal_docs"
  folder_display_name: "Legal Documents"
  file_types: [".pdf"]
  index_path: "faiss_index_legal"
query:
  placeholder: "e.g., What are the contract terms?"
  help_text: "Ask questions about legal documents"
prompts:
  system_prompt: "You are a legal document assistant..."
processing:
  chunk_size: 500
  chunk_overlap: 100
```

### Switching Between Domains

```bash
# Set environment variable
export CHATBOT_DOMAIN=legal

# Or use inline for specific commands
CHATBOT_DOMAIN=technical python cli.py --query "How do I configure the API?"

# Web interface with specific domain
CHATBOT_DOMAIN=financial streamlit run streamlit_app.py
```

## 🏗️ Architecture

### Project Structure

```
HRBot/
├── 📁 Core Application
│   ├── document_chatbot.py    # Main chatbot functionality (domain-agnostic)
│   ├── domain_config.py       # Domain configuration system
│   ├── async_hr_chatbot.py    # Async implementation
│   ├── config.py              # Configuration management
│   ├── validation.py          # Input validation & security
│   ├── cache.py               # Caching system
│   ├── chunking.py            # Smart document chunking
│   └── services.py            # Service layer with DI
├── 📁 Interfaces
│   ├── cli.py                 # Command line interface
│   ├── streamlit_app.py       # Web interface
│   ├── create_domain.py       # Domain generator tool
│   └── query.py               # Query utilities
├── 📁 Domain Configurations
│   ├── domains/hr.yaml        # HR domain configuration
│   ├── domains/legal.yaml     # Legal domain configuration
│   ├── domains/technical.yaml # Technical domain configuration
│   └── domains/financial.yaml # Financial domain configuration
├── 📁 Infrastructure
│   ├── Dockerfile             # Production container
│   ├── Dockerfile.dev         # Development container
│   ├── docker-compose.yml     # Multi-service setup
│   └── Makefile               # Development commands
├── 📁 Testing
│   ├── tests/unit/            # Unit tests
│   ├── tests/integration/     # Integration tests
│   └── tests/fixtures/        # Test fixtures
├── 📁 Configuration
│   ├── .env.example           # Environment template
│   ├── .pre-commit-config.yaml # Code quality hooks
│   ├── pyproject.toml         # Project configuration
│   ├── requirements.txt       # Production dependencies
│   └── requirements-dev.txt   # Development dependencies
└── 📁 Data
    ├── policies/              # HR policy documents
    ├── legal_docs/            # Legal documents
    ├── tech_docs/             # Technical documentation
    ├── financial_docs/        # Financial documents
    ├── faiss_index_hr/        # HR vector store index
    ├── faiss_index_legal/     # Legal vector store index
    └── cache/                 # Application cache
```

### Key Components

- **🎯 Domain System**: YAML-based configuration for different document types
- **🧠 Smart Chunking**: Context-aware document splitting for better retrieval
- **⚡ Caching Layer**: Multi-level caching (query results, embeddings)
- **🛡️ Security**: Input validation, sanitization, and threat detection
- **📊 Monitoring**: Comprehensive logging and performance metrics
- **🔧 Service Layer**: Dependency injection for better testability

## ⚙️ Configuration

All configuration is managed through environment variables and domain YAML files. See `.env.example` for all available options:

### Core Settings
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `CHATBOT_DOMAIN`: Active domain (hr, legal, technical, financial)
- `EMBEDDING_MODEL`: Embedding model (default: text-embedding-3-large)
- `LLM_MODEL`: Language model (default: gpt-4o)

### Performance Settings
- `CHUNK_SIZE`: Document chunk size (default: 500)
- `CHUNK_OVERLAP`: Chunk overlap (default: 100)
- `CACHE_ENABLED`: Enable caching (default: true)
- `MAX_CACHE_SIZE`: Maximum cache entries (default: 1000)

### Security Settings
- `MAX_QUERY_LENGTH`: Maximum query length (default: 1000)
- `ALLOWED_FILE_EXTENSIONS`: Allowed file types (default: .pdf)

## 🐳 Docker Deployment

### Production
```bash
docker build -t document-chatbot .
docker run -p 8501:8501 --env-file .env document-chatbot
```

### Development
```bash
make docker-dev  # Starts all services with hot reload
```

### Docker Compose Services
- `document-chatbot`: Production service
- `document-chatbot-dev`: Development with hot reload
- `jupyter`: Jupyter Lab for exploration

## ☁️ Streamlit Cloud Deployment

**✅ Ready for Streamlit Cloud!** The app is fully configured for cloud deployment.

### Quick Deploy to Streamlit Cloud:

1. **Push to GitHub:**
```bash
git push origin main
```

2. **Deploy on [share.streamlit.io](https://share.streamlit.io):**
   - Connect your GitHub repository
   - Set main file: `streamlit_app.py`
   - Add your `OPENAI_API_KEY` in app secrets

3. **Your app will be live at:** `https://your-app-name.streamlit.app`

📖 **Detailed deployment guide:** See [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md)

## 🧪 Testing & Quality

### Running Tests
```bash
make test              # All tests
make test-unit         # Unit tests only
make test-integration  # Integration tests only
make test-coverage     # Coverage report
```

### Code Quality
```bash
make lint              # Linting with flake8
make format            # Auto-format with black & isort
make type-check        # Type checking with mypy
make security-scan     # Security scan with bandit
make pre-commit        # Run all quality checks
```

### Pre-commit Hooks
```bash
pre-commit install     # Install hooks
pre-commit run --all-files  # Run manually
```

## 📈 Performance & Monitoring

### Benchmarking
```bash
make benchmark         # Run performance benchmarks
make load-test         # Load testing with Locust
```

### Monitoring Features
- Comprehensive logging with configurable levels
- Query performance metrics
- Cache hit/miss statistics
- Error tracking and alerting
- Resource usage monitoring

## 🔐 Security

### Input Validation
- SQL injection prevention
- XSS protection
- Path traversal prevention
- Content sanitization

### API Security
- Rate limiting (configurable)
- API key rotation support
- Audit logging
- Secure file handling

## 🚀 Production Deployment

### Pre-deployment Checklist
```bash
make pre-deploy        # Runs all production checks
```

### Environment Setup
1. Set production environment variables
2. Configure logging and monitoring
3. Set up backup procedures for vector indices
4. Configure rate limiting and security policies

### Scaling Considerations
- Horizontal scaling with multiple containers
- Shared cache layer (Redis recommended)
- Load balancing for high availability
- Database clustering for large document sets

## 📖 Document Types

The framework can process various document types across different domains:

### HR Documents
- 📋 Absence and leave policies
- 🏥 Health and safety guidelines
- ⚖️ Disciplinary and grievance procedures
- 🌍 Diversity and inclusion policies
- 🔒 Compliance and regulatory requirements

### Legal Documents
- ⚖️ Contract templates and agreements
- 📋 Legal policies and procedures
- 🔒 Compliance documentation
- 📄 Terms of service and privacy policies

### Technical Documentation
- 📚 API documentation and guides
- 🔧 System manuals and specifications
- 📋 Installation and configuration guides
- 🏗️ Architecture and design documents

### Financial Documents
- 💰 Expense and accounting policies
- 📊 Budget guidelines and procedures
- 🏦 Procurement and invoice processing
- 📈 Financial compliance materials

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Run tests: `make test`
4. Run quality checks: `make pre-commit`
5. Submit a pull request

### Development Setup
```bash
make setup-dev         # Complete development setup
make info              # Show environment information
```

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

- 📚 [Documentation](docs/)
- 🐛 [Issues](https://github.com/your-org/document-chatbot/issues)
- 💬 [Discussions](https://github.com/your-org/document-chatbot/discussions)

## 🏆 Acknowledgments

Built with:
- [LangChain](https://langchain.com) - LLM application framework
- [OpenAI](https://openai.com) - GPT-4 and embeddings
- [FAISS](https://faiss.ai) - Vector similarity search
- [Streamlit](https://streamlit.io) - Web interface