# HR Policy Chatbot

An intelligent chatbot that helps employees query company HR policies using natural language. The system uses Retrieval-Augmented Generation (RAG) to provide accurate, context-aware answers from HR policy documents.

## Features

- **PDF Document Processing**: Automatically processes HR policy PDFs and creates searchable embeddings
- **Semantic Search**: Uses FAISS vector store for fast, accurate document retrieval
- **Multiple Interfaces**: CLI, Streamlit web app, and Python API
- **Context-Aware Responses**: Answers are grounded in actual policy documents with source citations
- **OpenAI Integration**: Leverages GPT-4 for intelligent question answering

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd HRBot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

4. Build the search index from your policy documents:
```bash
python cli.py --index
```

## Usage

### Web Interface (Streamlit)

Launch the web interface:
```bash
streamlit run streamlit_app.py
```

Navigate to `http://localhost:8501` to use the interactive chatbot.

### Command Line Interface

Build the index:
```bash
python cli.py --index --folder ./policies
```

Ask a question:
```bash
python cli.py --query "What is the sick leave policy?"
```

Query with custom parameters:
```bash
python cli.py --query "How many vacation days do I get?" --k 5
```

### Python API

```python
from hr_chatbot import load_vectorstore, answer_query

# Load the vector store
vectorstore = load_vectorstore("faiss_index_hr")

# Ask a question
answer, sources = answer_query("What is the remote work policy?", vectorstore)
print(answer)
```

## Project Structure

```
HRBot/
├── hr_chatbot.py          # Core chatbot functionality
├── cli.py                 # Command line interface
├── streamlit_app.py       # Web interface
├── main.py               # Example usage script
├── query.py              # Query utilities
├── embedding.py          # Embedding utilities
├── policies/             # HR policy PDF documents
├── faiss_index_hr/       # Generated FAISS index files
├── requirements.txt      # Python dependencies
└── Dockerfile           # Docker configuration
```

## Configuration

### Document Processing
- **Chunk Size**: 500 tokens (configurable)
- **Chunk Overlap**: 100 tokens (configurable)
- **Embedding Model**: text-embedding-3-large
- **LLM Model**: gpt-4o

### Customization

You can customize the behavior by modifying parameters in `hr_chatbot.py`:

- `chunk_size`: Size of document chunks for processing
- `chunk_overlap`: Overlap between chunks
- `embedding_model_name`: OpenAI embedding model
- `llm_model_name`: OpenAI language model
- `k`: Number of source documents to retrieve

## Docker Support

Build and run with Docker:
```bash
docker build -t hr-chatbot .
docker run -e OPENAI_API_KEY=your-key -p 8501:8501 hr-chatbot
```

## Policy Documents

The system processes PDF documents from the `policies/` folder, including:
- Absence procedures and pay policies
- Holiday and leave policies
- Disciplinary and grievance procedures
- Health and safety guidelines
- Compliance and regulatory policies
- And more...

## API Key Security

Never commit your OpenAI API key to version control. Use environment variables or secure secret management systems in production.

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]