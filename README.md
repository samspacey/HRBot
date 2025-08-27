# Document Chatbot Framework

Note: This repository has been simplified to a minimal Streamlit proof-of-concept. Many advanced features mentioned below (multi-domain, CLI, async, upload workflows) have been removed to keep the demo focused. To run: place a few PDFs in `policies/` and start `streamlit_app.py`. The app will build the index automatically on first run.

An enterprise-grade intelligent chatbot framework that enables organizations to create domain-specific document query assistants using natural language. The system uses Retrieval-Augmented Generation (RAG) architecture with advanced document processing, caching, and validation to provide accurate, context-aware answers from any document collection.

**🎯 Multi-Domain Support:** Easily create chatbots for HR policies, legal documents, technical documentation, financial procedures, and more with simple YAML configuration files.

## ✨ Features

- **📚 Interactive Document Upload**: Upload your own documents and chat with them instantly
- **🎯 Multi-Domain Framework**: Create chatbots for any document type (HR, Legal, Technical, Financial, etc.)
- **⚙️ YAML Configuration**: Simple configuration files for different domains and use cases
- **📄 Multi-Format Support**: PDF, TXT, Markdown, and Word documents
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

3. **🔑 Add Your OpenAI API Key:**
Export it in your shell so the app can read it:
```
export OPENAI_API_KEY=sk-your-real-key
```

4. **📚 Add Documents:**
Place a few PDF files into the `policies/` folder. The app builds the FAISS index automatically on first run.

4. **🚀 Start the App:**
```bash
make serve  # Opens web interface at http://localhost:8501
```

**That's it!** 🎉 Your document chatbot is now running!

### 📚 Interactive Uploads
Removed in this POC to keep the demo simple. Uploading via the UI may return in a future iteration.

### 🎯 Multi-Domain and Generators
Removed in this POC. The demo targets a single `policies/` folder for simplicity.

### Alternative Ways to Run
CLI and domain switching have been removed in this POC.

**Docker (if you prefer containers):**
```bash
docker build -t document-chatbot .
docker run -p 8501:8501 --env-file .env document-chatbot
```

**Development Mode (with hot reload):** Use `make serve` locally or build a Docker image with `make docker-build`.
The app reads `OPENAI_API_KEY` from your environment; no `.env` file is required.

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
Useful commands:
```bash
make help         # Show available commands
make setup-dev    # Install deps and create .env
make format       # Black + isort
make lint         # Flake8
make type-check   # mypy
make serve        # Start Streamlit UI
```

### Web Interface (Streamlit)

```bash
make serve
# Or directly: streamlit run streamlit_app.py
```

Navigate to `http://localhost:8501` for the interactive chatbot interface.

### Command Line Interface
Not available in this POC. Use the Streamlit UI.

### Python API
If needed, see `simple_rag.py` for minimal functions to load/build the index and answer queries.

### Async Usage
Not applicable for this POC.

## 🎯 Domain Configuration
Removed in this POC.

## 🏗️ Architecture

### Project Structure

```
HRBot/
├── 📁 Core Application
│   ├── simple_rag.py          # Minimal RAG helper
│   └── config.py              # Configuration management
├── 📁 Interfaces
│   └── streamlit_app.py       # Web interface
├── 📁 Infrastructure
│   ├── Dockerfile             # Production container
│   ├── Dockerfile.dev         # Development container
│   ├── docker-compose.yml     # Multi-service setup
│   └── Makefile               # Development commands
├── 📁 Configuration
│   ├── .env.example           # Environment template
│   ├── .pre-commit-config.yaml # Code quality hooks
│   ├── pyproject.toml         # Project configuration
│   ├── requirements.txt       # Production dependencies
│   └── requirements-dev.txt   # Development dependencies
└── 📁 Data
    ├── policies/              # Demo PDFs
    └── faiss_index_hr/        # Vector index (generated)
```

### Key Components
- Minimal RAG using FAISS, OpenAI embeddings, and ChatGPT via LangChain
- Streamlit UI for simple Q&A over local PDFs

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
Use local `make serve` or Docker build/run commands above.

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
