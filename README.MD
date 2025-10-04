# RAG-QA: Retrieval Augmented Generation Question Answering

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A containerized, open-source framework for document-based Question Answering using Retrieval Augmented Generation (RAG). Upload your documents and get intelligent, context-aware answers powered by state-of-the-art language models.

## 🚀 Features

- **Multi-LLM Support**: Integration with Google Gemini Pro and OpenAI models (GPT-4, GPT-3.5-turbo)
- **Advanced Embeddings**: Uses all-MiniLM-L6-v2 model for high-quality document embeddings
- **Vector Database**: Chroma DB for efficient similarity search and retrieval
- **Modern UI**: Clean, intuitive Streamlit frontend
- **Robust API**: FastAPI backend for scalable deployment
- **Containerized**: Docker Compose for easy setup and deployment
- **Document Support**: PDF document processing and indexing
- **Privacy-Focused**: Run entirely locally with your own API keys

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **LLMs**: Google Gemini Pro, OpenAI GPT-4/GPT-3.5-turbo
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Database**: Chroma DB
- **Containerization**: Docker & Docker Compose
- **Language**: Python 3.8+

## 📋 Prerequisites

- Docker and Docker Compose installed
- API key from either:
  - Google AI (for Gemini Pro)
  - OpenAI (for GPT models)

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/sir-pwnsalot/RAG-QA.git
cd RAG-QA
```

### 2. Configure Environment
```bash
cp .env.example .env
```

Edit the `.env` file with your API credentials:
```env
# Choose your LLM provider
GOOGLE_API_KEY=your_google_api_key_here
# OR
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Customize model settings
MODEL_NAME=gemini-pro  # or gpt-4, gpt-3.5-turbo
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./chroma_db
```

### 3. Launch the Application
```bash
docker-compose up -d
```

### 4. Access the Interface
Open your browser and navigate to:
- **Frontend**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs

## 💡 Usage

### Web Interface
1. **Upload Document**: Use the sidebar to upload a PDF document
2. **Wait for Processing**: The system will index your document automatically
3. **Ask Questions**: Type your questions in the chat interface
4. **Get Answers**: Receive context-aware responses based on your document

### API Usage
The FastAPI backend provides programmatic access:

```python
import requests

# Upload document
files = {'file': open('document.pdf', 'rb')}
response = requests.post('http://localhost:8000/upload', files=files)

# Ask question
payload = {'question': 'What is the main topic of this document?'}
response = requests.post('http://localhost:8000/ask', json=payload)
answer = response.json()['answer']
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google AI API key | None |
| `OPENAI_API_KEY` | OpenAI API key | None |
| `MODEL_NAME` | LLM model to use | `gemini-pro` |
| `EMBEDDING_MODEL` | Embedding model | `all-MiniLM-L6-v2` |
| `CHROMA_PERSIST_DIR` | Chroma DB storage path | `./chroma_db` |
| `MAX_TOKENS` | Maximum response tokens | `1000` |
| `TEMPERATURE` | Model temperature | `0.1` |

### Supported Models

**Google Gemini:**
- `gemini-pro`
- `gemini-pro-vision`

**OpenAI:**
- `gpt-4`
- `gpt-4-turbo`
- `gpt-3.5-turbo`

## 📁 Project Structure

```
RAG-QA/
├── src/
│   ├── backend/
│   │   ├── main.py              # FastAPI application
│   │   ├── models/              # Data models
│   │   ├── services/            # Business logic
│   │   └── utils/               # Utility functions
│   ├── frontend/
│   │   ├── app.py               # Streamlit application
│   │   ├── components/          # UI components
│   │   └── utils/               # Frontend utilities
│   └── shared/
│       ├── config.py            # Configuration management
│       ├── embeddings.py        # Embedding utilities
│       └── vector_store.py      # Vector database operations
├── docker-compose.yml           # Docker services
├── Dockerfile.backend           # Backend container
├── Dockerfile.frontend          # Frontend container
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
└── README.md                    # This file
```

## 🔍 How It Works

1. **Document Processing**: PDF documents are parsed and split into chunks
2. **Embedding Generation**: Each chunk is converted to vector embeddings using sentence-transformers
3. **Vector Storage**: Embeddings are stored in Chroma DB for fast similarity search
4. **Query Processing**: User questions are embedded and used to retrieve relevant document chunks
5. **Answer Generation**: Retrieved context + question are sent to the LLM for final answer generation

## 🚀 Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run backend
cd src/backend && uvicorn main:app --reload

# Run frontend (in another terminal)
cd src/frontend && streamlit run app.py
```

### Production Deployment
For production, consider:
- Using a reverse proxy (nginx)
- Implementing authentication
- Setting up SSL certificates
- Configuring persistent storage
- Implementing rate limiting

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Issues & Support

If you encounter any issues or need support:
1. Check existing [Issues](https://github.com/sir-pwnsalot/RAG-QA/issues)
2. Create a new issue with detailed description
3. Include error logs and system information

## 🔮 Roadmap

- [ ] Support for more document formats (DOCX, TXT, HTML)
- [ ] Multi-document querying
- [ ] Advanced retrieval strategies
- [ ] Integration with more LLM providers
- [ ] Chat history and session management
- [ ] Advanced document preprocessing
- [ ] Deployment templates for cloud platforms

## 👨‍💻 Author

**sir-pwnsalot**
- GitHub: [@sir-pwnsalot](https://github.com/sir-pwnsalot)
- LinkedIn: [Connect with me](https://linkedin.com/in/sir-pwnsalot)

## 📊 Stats

![GitHub stars](https://img.shields.io/github/stars/sir-pwnsalot/RAG-QA)
![GitHub forks](https://img.shields.io/github/forks/sir-pwnsalot/RAG-QA)
![GitHub issues](https://img.shields.io/github/issues/sir-pwnsalot/RAG-QA)

---

⭐ **Star this repository if you found it helpful!**