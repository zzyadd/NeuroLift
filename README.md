# 💬 RAG-Powered Chat Server with Memory & Supabase Logging

This project is a FastAPI-based server for an advanced chat assistant that uses RAG (Retrieval-Augmented Generation), conversation memory with Redis, and Supabase for persistent storage. It supports English and Arabic and integrates ChromaDB for document retrieval and SentenceTransformers for embedding.

## 🚀 Features

- 🔥 RAG pipeline using ChromaDB and `jinaai/jina-clip-v2`
- 💾 Short- and long-term memory using Redis
- 📚 Supabase integration for chat and conversation logs
- 🌐 Arabic & English language detection
- 🤖 Automatic model selection and categorization
- 📝 Summarization of long conversation histories
- 🔗 Optional ngrok tunneling for public exposure
- 🧠 Embedding via `SentenceTransformer` (local)
- 📎 Session management & caching

## 🛠️ Technologies Used

- [FastAPI](https://fastapi.tiangolo.com/)
- [Redis](https://redis.io/)
- [ChromaDB](https://www.trychroma.com/)
- [SentenceTransformers](https://www.sbert.net/)
- [Supabase](https://supabase.com/)
- [Ollama](https://ollama.com/)
- [pyngrok](https://github.com/alexdlaird/pyngrok)
- [LangDetect](https://pypi.org/project/langdetect/)

## 📦 Installation

### 1. Clone the repo

```bash
git clone https://github.com/your-username/chat-rag-server.git
cd chat-rag-server
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
```
3. Configure .env
Create a .env file in the root directory with the following:
```
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=smollm2:latest
ARABIC_MODEL=prakasharyan/qwen-arabic:latest
EMBEDDINGS_MODEL=jinaai/jina-clip-v2
REDIS_URL=redis://localhost:6379/0
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-service-role-key
USE_NGROK=true
NGROK_AUTHTOKEN=your-ngrok-token
```

🧪 Running the Server
bash
Copy
Edit
python server.py

📌 Notes
Embeddings and model inference are performed locally using CPU or CUDA.

Documents used for retrieval are stored in ChromaDB (chroma_data_godic).

Conversations are cached, summarized, and reused for efficient performance.

Supabase stores all conversation and user metadata for long-term analytics.

🧠 Memory Logic
Memory TTL = 1 hour (by default)

Long-term memory TTL = 30 days

Memories include embeddings for semantic recall using cosine similarity

🛡️ Authentication
Supabase JWT-based Bearer token required for /api/user-chats, /api/chats, and /api/chat-with-db
