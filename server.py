#server.py

import asyncio
import os
import time
import uuid
import json
import re
import logging
import httpx
import numpy as np
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Any, Tuple, Protocol
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from redis import asyncio as aioredis
from contextlib import asynccontextmanager
from pyngrok import ngrok  # Ngrok import
from supabase import create_client, Client
from dotenv import load_dotenv
from supabase import Client as SupabaseClient
import chromadb
from sentence_transformers import SentenceTransformer
import torch

load_dotenv(".env")

# ------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------
logger = logging.getLogger("chat_debug")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# ------------------------------------------------------------
# Settings and Configuration
# ------------------------------------------------------------
class Settings:
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://192.168.1.117:11434")
    ARABIC_MODEL: str = os.getenv("ARABIC_MODEL", "prakasharyan/qwen-arabic:latest")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "smollm2:latest")
    CATEGORIZER_MODEL: str = os.getenv("CATEGORIZER_MODEL", "granite3.2:2b")
    MEMORY_TTL: int = int(os.getenv("MEMORY_TTL", "3600"))
    LONG_TERM_MEMORY_TTL: int = int(os.getenv("LONG_TERM_MEMORY_TTL", "2592000"))
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CLEANUP_INTERVAL: int = int(os.getenv("CLEANUP_INTERVAL", "60"))
    MEMORY_CONTEXT_SIZE: int = int(os.getenv("MEMORY_CONTEXT_SIZE", "5"))
    EMBEDDINGS_MODEL: str = os.getenv("EMBEDDINGS_MODEL", "jinaai/jina-clip-v2")  # Changed to jina-clip-v2
    ARABIC_EMBEDDINGS_MODEL: Optional[str] = os.getenv("ARABIC_EMBEDDINGS_MODEL", "jinaai/jina-clip-v2")  # Changed to jina-clip-v2
    ENGLISH_MODEL_MAPPING: Dict[str, str] = {
        "summarization": os.getenv("MODEL_SUMMARIZATION", "llama3.2:1b"),
        "information_retrieval": os.getenv("MODEL_INFORMATION_RETRIEVAL", "llama3.2:1b"),
        "comparison": os.getenv("MODEL_COMPARISON", "llama3.2:1b"),
        "extraction": os.getenv("MODEL_EXTRACTION", "smollm2:latest"),
        "explanation": os.getenv("MODEL_EXPLANATION", "smollm2:latest"),
        "general": os.getenv("MODEL_GENERAL", "smollm2:latest"),
    }
    SUMMARIZATION_MODEL: str = os.getenv("SUMMARIZATION_MODEL", "llama3.2:1b")
    SUMMARY_TOKEN_LIMIT: int = int(os.getenv("SUMMARY_TOKEN_LIMIT", "500"))
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "https://xadjoggbirotiqenzxcb.supabase.co")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhhZGpvZ2diaXJvdGlxZW56eGNiIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MDc1NTI4MCwiZXhwIjoyMDU2MzMxMjgwfQ.zDSbRjMT-Y7Tmb2QaiKipfarOSy_PGvuGI132tLLVD4")

@lru_cache()
def get_settings() -> Settings:
    return Settings()

# ------------------------------------------------------------
# Domain Models
# ------------------------------------------------------------
class PromptRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    conversation_id: Optional[str] = None
    stream: bool = False
    use_memory: bool = True

class LLMResponse(BaseModel):
    response: str
    conversation_id: str
    category: str
    model_used: str
    processing_time: float = Field(default=0.0)
    memories_used: Optional[List[str]] = None

class Memory(BaseModel):
    id: str
    content: str
    embedding: List[float]
    timestamp: str
    source_conversation_id: str
    importance_score: float

class ChatRecord(BaseModel):
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    thread_id: Optional[str] = None
    title: str
    user_id: Optional[str] = None  # Make optional since it's auto-generated

class ConversationRecord(BaseModel):
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    chat_id: int
    type: str  # 'user' or 'assistant'
    content: str
    time: Optional[float] = Field(default=None)
    reference: Optional[str] = Field(default=None)  # Add reference field
    category: Optional[str] = Field(default=None)  # Add category field

# ------------------------------------------------------------
# Utility: Arabic Normalization (SRP)
# ------------------------------------------------------------
def normalize_arabic(text: str) -> str:
    """Normalize Arabic text: remove diacritics and standardize letters."""
    diacritics_pattern = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(diacritics_pattern, '', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'[يى]', 'ي', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'\ـ', '', text)
    return ' '.join(text.split())

# ------------------------------------------------------------
# Interfaces (Protocols)
# ------------------------------------------------------------
class IConversationRepository(Protocol):
    async def get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        ...
    async def save_history(self, conversation_id: str, history: List[Dict[str, str]]) -> None:
        ...
    async def delete_conversation(self, conversation_id: str) -> None:
        ...

class IMemoryService(Protocol):
    async def store_memory(self, content: str, conversation_id: str, language: str) -> str:
        ...
    async def search_memories(self, query: str, language: str, limit: int,
                              conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        ...
# ------------------------------------------------------------
# Services: Language Detection (SRP)
# ------------------------------------------------------------
class LanguageDetector:
    def __init__(self):
        try:
            import langdetect
            from langdetect import DetectorFactory
            DetectorFactory.seed = 0
            self.langdetect = langdetect
            self.is_available = True
        except ImportError:
            self.is_available = False

    def detect(self, text: str) -> str:
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        if len(text.strip()) < 3:
            return "english"
        if self.is_available:
            try:
                lang_code = self.langdetect.detect(text)
                return "arabic" if lang_code == "ar" else "english"
            except Exception as e:
                logger.error("Error in language detection: %s", e)
        return "arabic" if re.search(r'[\u0600-\u06FF]', text) else "english"

# ------------------------------------------------------------
# Persistence: Conversation Repository using Redis (SRP)
# ------------------------------------------------------------

class SupabaseRepository:
    def __init__(self, settings: Settings):
        self.supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

    async def create_chat(self, chat: ChatRecord) -> Dict[str, Any]:
        """Create a new chat thread"""
        data = {
            "thread_id": chat.thread_id or str(uuid.uuid4()),
            "title": chat.title,
    }
    # Only include user_id if explicitly provided
        if chat.user_id:
            data["user_id"] = chat.user_id

        result = self.supabase.table('chat').insert(data).execute()
        return result.data[0] if result.data else {}

    async def get_chat(self, chat_id: int) -> Dict[str, Any]:
        """Get chat by ID"""
        result = self.supabase.table('chat').select('*').eq('id', chat_id).execute()
        return result.data[0] if result.data else {}

    async def get_user_chats(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all chats for a user"""
        result = self.supabase.table('chat').select('*').eq('user_id', user_id).order('created_at', desc=True).execute()
        return result.data if result.data else []

    async def add_conversation_message(self, message: ConversationRecord) -> Dict[str, Any]:
        """Add a message to a conversation"""
        data = {
            "chat_id": message.chat_id,
            "type": message.type,
            "content": message.content,
            "time": message.time,
            "reference": message.reference  # Use the reference from the message object
        }
        
        # Add category if it exists
        if message.category:
            data["category"] = message.category
            
        result = self.supabase.table('conversation').insert(data).execute()
        return result.data[0] if result.data else {}

    async def get_conversation_history(self, chat_id: int) -> List[Dict[str, Any]]:
        """Get conversation history for a chat"""
        result = self.supabase.table('conversation').select('*').eq('chat_id', chat_id).order('created_at').execute()
        return result.data if result.data else []

    async def get_all_chats_by_thread_id(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get chats by thread_id"""
        result = self.supabase.table('chat').select('*').eq('thread_id', thread_id).execute()
        return result.data if result.data else []

    async def get_chat_by_thread_id(self, thread_id: str) -> Dict[str, Any]:
        """Get chat by thread_id"""
        result = self.supabase.table('chat').select('*').eq('thread_id', thread_id).single().execute()
        return result.data if result.data else {}

class RedisConversationRepository(IConversationRepository):
    def __init__(self, redis_client: aioredis.Redis, settings: Settings):
        self.redis = redis_client
        self.settings = settings

    async def get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        raw_data = await self.redis.get(f"conv:{conversation_id}")
        if not raw_data:
            logger.debug("No history found for %s", conversation_id)
            return []
        try:
            history = json.loads(raw_data)
            return history
        except Exception as e:
            logger.error("Error decoding conversation history: %s", e)
            return []

    async def save_history(self, conversation_id: str, history: List[Dict[str, str]]) -> None:
        for msg in history:
            if "timestamp" not in msg:
                msg["timestamp"] = datetime.now().isoformat()
        await self.redis.set(f"conv:{conversation_id}", json.dumps(history))
        await self.redis.expire(f"conv:{conversation_id}", self.settings.MEMORY_TTL)
        await self.redis.set(f"last_access:{conversation_id}", datetime.now().isoformat())
        await self.redis.expire(f"last_access:{conversation_id}", self.settings.MEMORY_TTL)
        logger.debug("Saved conversation history for %s", conversation_id)

    async def delete_conversation(self, conversation_id: str) -> None:
        await self.redis.delete(f"conv:{conversation_id}")
        await self.redis.delete(f"last_access:{conversation_id}")
        logger.info("Deleted conversation %s", conversation_id)

# ------------------------------------------------------------
# Services: Embedding Service using SentenceTransformer (SRP)
# ------------------------------------------------------------
class SentenceTransformerEmbeddingService:
    def __init__(self, settings: Settings, http_client: httpx.AsyncClient):
        self.settings = settings
        self.http_client = http_client
        # Initialize the model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading SentenceTransformer model on {device}")
        self.model = SentenceTransformer("jinaai/jina-clip-v2", trust_remote_code=True).to(device)
        self.device = device

    async def get_embedding(self, text: str, language: str = "english") -> List[float]:
        """Get embeddings using SentenceTransformer model directly"""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                logger.debug(f"Generating embedding for text: {text[:50]}... (language: {language})")
                # Process in a separate thread to avoid blocking the event loop
                embedding = await asyncio.to_thread(self._generate_embedding, text)
                logger.debug(f"Embedding successfully generated for text: {text[:50]}...")
                return embedding
            except Exception as e:
                logger.error(f"Error generating embedding (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.warning(f"Retrying embedding generation in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)

        # If all retries fail, return a zero vector as a fallback
        logger.error("All attempts to generate embedding failed. Returning a zero vector as a fallback.")
        return [0.0] * 768

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using the model (runs in a separate thread)"""
        with torch.no_grad():
            embedding = self.model.encode(text)
            # Convert to list and normalize if needed
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy().tolist()
            elif isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            return embedding

# ------------------------------------------------------------
# Services: Redis Session and Cache Manager (SRP)
# ------------------------------------------------------------
class RedisSessionManager:
    def __init__(self, redis_client: aioredis.Redis, settings: Settings):
        self.redis = redis_client
        self.settings = settings
        self.cache_ttl = 3600  # Default 1 hour cache TTL

    async def get_session(self, user_id: str) -> Dict[str, Any]:
        """Get user session data"""
        session_data = await self.redis.get(f"session:{user_id}")
        if not session_data:
            return {}
        try:
            return json.loads(session_data)
        except Exception as e:
            logger.error(f"Error decoding session data: {e}")
            return {}

    async def set_session(self, user_id: str, session_data: Dict[str, Any]) -> None:
        """Set user session data"""
        await self.redis.set(f"session:{user_id}", json.dumps(session_data))
        await self.redis.expire(f"session:{user_id}", self.settings.MEMORY_TTL)

    async def delete_session(self, user_id: str) -> None:
        """Delete user session"""
        await self.redis.delete(f"session:{user_id}")

    async def cache_set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a cached value with TTL"""
        ttl = ttl or self.cache_ttl
        await self.redis.set(f"cache:{key}", json.dumps(value))
        await self.redis.expire(f"cache:{key}", ttl)

    async def cache_get(self, key: str) -> Optional[Any]:
        """Get a cached value"""
        cached_data = await self.redis.get(f"cache:{key}")
        if not cached_data:
            return None
        try:
            return json.loads(cached_data)
        except Exception as e:
            logger.error(f"Error decoding cached data: {e}")
            return None

    async def cache_delete(self, key: str) -> None:
        """Delete a cached value"""
        await self.redis.delete(f"cache:{key}")

# ------------------------------------------------------------
# Services: Memory Service using Redis (SRP)
# ------------------------------------------------------------
class RedisMemoryService(IMemoryService):
    def __init__(self, redis_client: aioredis.Redis, embedding_service: SentenceTransformerEmbeddingService):
        self.redis = redis_client
        self.embedding_service = embedding_service
        self.settings = get_settings()

    async def store_memory(self, content: str, conversation_id: str, language: str) -> str:
        normalized_content = normalize_arabic(content) if language.lower() == "arabic" else content
        embedding = await self.embedding_service.get_embedding(normalized_content, language=language)
        if language.lower() == "arabic":
            word_count = len(normalized_content.split())
            importance_score = min(word_count / 20, 1.0)
        else:
            importance_score = min(len(content) / 100, 1.0)
        memory_id = str(uuid.uuid4())
        memory = Memory(
            id=memory_id,
            content=content,
            embedding=embedding,
            timestamp=datetime.now().isoformat(),
            source_conversation_id=conversation_id,
            importance_score=importance_score
        )
        if language.lower() == "arabic":
            memory_key = f"arabic_memory:{memory_id}"
            conv_index = f"conversation_memories_arabic:{conversation_id}"
        else:
            memory_key = f"english_memory:{memory_id}"
            conv_index = f"conversation_memories_english:{conversation_id}"
        await self.redis.set(memory_key, memory.json(), ex=self.settings.LONG_TERM_MEMORY_TTL)
        await self.redis.sadd(conv_index, memory_id)
        return memory_id

    async def search_memories(self, query: str, language: str, limit: int = 5,
                              conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        normalized_query = normalize_arabic(query) if language.lower() == "arabic" else query
        query_embedding = await self.embedding_service.get_embedding(normalized_query, language=language)
        key_prefix = "arabic_memory:" if language.lower() == "arabic" else "english_memory:"
        memories_with_scores = []
        all_keys = await self.redis.keys(f"{key_prefix}*")
        for key in all_keys:
            mem_data = await self.redis.get(key)
            if not mem_data:
                continue
            memory = Memory.parse_raw(mem_data)
            similarity = self._cosine_similarity(query_embedding, memory.embedding)
            memories_with_scores.append({"memory": memory, "score": similarity})
        memories_with_scores.sort(key=lambda x: x["score"], reverse=True)
        top_memories = memories_with_scores[:limit]
        return [{"id": m["memory"].id, "content": m["memory"].content, "score": m["score"]} for m in top_memories]

    async def summarize_history(self, conversation_id: str, language: str, token_limit: int = 500) -> str:
        """Summarize conversation history using summarization model, respecting the token limit"""
        # Get full conversation history
        full_history = await self.redis.get(f"conv:{conversation_id}")
        if not full_history:
            return ""

        try:
            history = json.loads(full_history)
            # Filter out system messages
            user_assistant_msgs = [msg for msg in history if msg.get("role") in ["user", "assistant"]]

            if len(user_assistant_msgs) <= 2:  # Not enough history to summarize
                return ""

            # Create prompt for summarization with explicit token limit instruction
            if language.lower() == "arabic":
                summarization_prompt = (
                    f"أنت مساعد مفيد. قم بتلخيص المحادثة التالية في {token_limit} رمز فقط. "
                    f"ركز على المعلومات المهمة والسياق الرئيسي فقط. "
                    f"يجب أن يكون تلخيصك موجزًا وشاملًا في نفس الوقت، ولا يتجاوز {token_limit} رمز:\n\n"
                    + "\n".join([f"{msg['role']}: {msg['content']}" for msg in user_assistant_msgs])
                )
            else:
                summarization_prompt = (
                    f"You are a helpful assistant. Summarize the following conversation in exactly {token_limit} tokens or less. "
                    f"Focus only on key information and main context. "
                    f"Your summary must be concise yet comprehensive, and must not exceed {token_limit} tokens:\n\n"
                    + "\n".join([f"{msg['role']}: {msg['content']}" for msg in user_assistant_msgs])
                )

            # Use summarization model to generate summary
            url = f"{self.settings.OLLAMA_BASE_URL}/api/chat"
            model = self.settings.SUMMARIZATION_MODEL

            # Add parameters to control token generation
            parameters = {
                "max_tokens": token_limit,
                "temperature": 0.3,  # Lower temperature for more focused summary
                "top_p": 0.9,        # Slightly lower top_p for more precise output
                "stop": ["."*4]      # Stop generating after 4 dots in a row (indicates end)
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": summarization_prompt}],
                        "stream": False,
                        "options": parameters  # Pass parameters to control output length
                    }
                )
                response.raise_for_status()
                summary = response.json()["message"]["content"]

                # Store this summary for future use
                await self.redis.set(
                    f"summary:{conversation_id}",
                    summary,
                    ex=self.settings.MEMORY_TTL
                )
                return summary

        except Exception as e:
            logger.error(f"Error summarizing conversation history: {e}")
            return ""

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        if magnitude1 * magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)

# ------------------------------------------------------------
# Business Logic: Conversation Service (SRP)
# ------------------------------------------------------------
class ConversationService:
    def __init__(self, repository: IConversationRepository):
        self.repository = repository

    async def get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        return await self.repository.get_history(conversation_id)

    async def save_history(self, conversation_id: str, history: List[Dict[str, str]]) -> None:
        await self.repository.save_history(conversation_id, history)

    async def delete_conversation(self, conversation_id: str) -> None:
        await self.repository.delete_conversation(conversation_id)

# ------------------------------------------------------------
# Business Logic: Model Selector (SRP)
# ------------------------------------------------------------
log2_file = "selected_model_category.txt"

class ModelSelector:
    def __init__(self, settings: Settings, language_detector: Optional[LanguageDetector] = None):
        self.settings = settings
        self.language_detector = language_detector or LanguageDetector()

    async def select_model(self, text: str, client: httpx.AsyncClient, override_model: Optional[str] = None) -> Tuple[str, str]:
        if override_model:
            return override_model, "custom"
        language = self.language_detector.detect(text)
        if language == "arabic":
            return self.settings.ARABIC_MODEL, "arabic"
        category = await self.categorize_english_query(text, client)
        model = self.settings.ENGLISH_MODEL_MAPPING.get(category, self.settings.DEFAULT_MODEL)
        with open(log2_file, "a") as f:
            f.write(f"using model:{model} \n")
        return model, category

    async def categorize_english_query(self, text: str, client: httpx.AsyncClient) -> str:
        system_prompt = (
            "You are a query classifier. Categorize the user's input into exactly ONE category:\n"
            "- summarization\n- information_retrieval\n- comparison\n- extraction\n- explanation\n- general\n"
            "Respond with ONLY the category name."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        try:
            category = await self.query_ollama(messages, self.settings.CATEGORIZER_MODEL, client, stream=False)
            category = category.strip().lower()
            with open(log2_file, "a") as f:
                f.write(f"selected category: { category if category in self.settings.ENGLISH_MODEL_MAPPING else 'general'}\n")
            return category if category in self.settings.ENGLISH_MODEL_MAPPING else "general"
        except Exception as e:
            logger.warning("Error categorizing query: %s", e)
            return "general"

    @staticmethod
    async def query_ollama(messages: List[Dict[str, str]], model: str,
                            client: httpx.AsyncClient, stream: bool = False) -> Any:
        url = f"{get_settings().OLLAMA_BASE_URL}/api/chat"
        payload = {"model": model, "messages": messages, "stream": stream}

        try:
            response = await client.post(url, json=payload, timeout=60.0)
            response.raise_for_status()
            if stream:
                return response
            result = response.json()["message"]["content"]
            return result.decode("utf-8") if isinstance(result, bytes) else result
        except Exception as e:
            logger.error("Ollama API error: %s", e)
            raise HTTPException(status_code=500, detail=f"Ollama Error: {e}")

# ------------------------------------------------------------
# Business Logic: Chat Controller (SRP)
# ------------------------------------------------------------
# Initialize ChromaDB client and collection
log_file = "reference_log.txt"
chroma_client = chromadb.PersistentClient(path="chroma_data_godic")
chroma_collection = chroma_client.get_collection(name="jinaai")

def get_query_embeddings(user_input):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("jinaai/jina-clip-v2", trust_remote_code=True).to(device)
    response = model.encode(user_input, show_progress_bar=False)
    return response.tolist()

def query_chroma_db(collection, query_embeddings, n_results=1):
    results = collection.query(query_embeddings=[query_embeddings], n_results=n_results)
    if results and 'documents' in results and results['documents'][0]:
        # Also return metadata for reference
        doc_id = results['ids'][0][0]
        with open(log_file, "a") as f:
            f.write(f"{doc_id}\n")
        doc = results['documents'][0][0]
        meta = results['metadatas'][0][0] if 'metadatas' in results and results['metadatas'][0] else {}
        print(f"doc_id: {doc_id}")
        print(f"doc: {doc}")
        print(f"meta: {meta}")
        return doc_id, doc, meta
    return None, None, None

class ChatController:
    def __init__(self,
                 conv_service: ConversationService,
                 memory_service: IMemoryService,
                 http_client: httpx.AsyncClient,
                 model_selector: ModelSelector):
        self.conv_service = conv_service
        self.memory_service = memory_service
        self.http_client = http_client
        self.model_selector = model_selector
        self.settings = get_settings()

    def format_response(self, text: str) -> str:
        """Format the response with HTML styling."""
        # Format bold text
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

        # Format bullet points
        text = re.sub(r'\* (.*?)(?=\n|$)', r'<li>\1</li>', text)

        # Wrap bullet point lists in ul tags
        if '<li>' in text:
            text = f"<ul class='list-disc list-inside'>{text}</ul>"

        # Add paragraph breaks
        text = re.sub(r'\n\n+', '</p><p>', text)
        if text:
            text = f"<p>{text}</p>"

        # Replace remaining newlines with breaks
        text = text.replace('\n', '<br>')

        return text

    async def process_chat(self, req: PromptRequest, background_tasks: BackgroundTasks) -> LLMResponse:
        start_time = time.time()
        conv_id = req.conversation_id or str(uuid.uuid4())
        memories_used = []
        language = self.model_selector.language_detector.detect(req.prompt)
        filtered_history = await self.get_filtered_history(conv_id, language)
        
        # Check if history is long enough to warrant summarization (more than 10 messages)
        if len(filtered_history) > 10:
            # Summarize history in background to avoid delaying response
            background_tasks.add_task(
                self.memory_service.summarize_history,
                conv_id,
                language,
                self.settings.SUMMARY_TOKEN_LIMIT
            )
            
            # For immediate use, get existing summary if available
            summary = await self.redis.get(f"summary:{conv_id}")
            if summary:
                # Replace older history with summary, keeping only the most recent 5 exchanges
                recent_history = filtered_history[-10:]
                filtered_history = [
                    {"role": "system", "content": f"Previous conversation summary: {summary}", "timestamp": datetime.now().isoformat()}
                ] + recent_history

        # --- RAG: Retrieve relevant document from ChromaDB and add to context ---
        rag_doc, rag_meta = None, None
        try:
            query_emb = get_query_embeddings(req.prompt)
            rag_doc, rag_meta = query_chroma_db(chroma_collection, query_emb, n_results=1)
            if rag_doc:
                # Add retrieved doc as system context
                ref_title = rag_meta.get("book_title", "original source") if rag_meta else "original source"
                filtered_history.insert(0, {
                    "role": "system",
                    "content": f"Reference from {ref_title}:\n{rag_doc}",
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as e:
            # If RAG fails, continue without it
            pass

        model, category = await self.model_selector.select_model(
            req.prompt,
            self.http_client,
            req.model
        )

        # Retrieve relevant memories if enabled
        if req.use_memory:
            relevant_memories = await self.memory_service.search_memories(
                req.prompt,
                language,
                self.settings.MEMORY_CONTEXT_SIZE,
                conv_id
            )
            memories_used = [m["content"] for m in relevant_memories]

            # Add memories to system context
            memory_context = "\n\n".join(memories_used)
            filtered_history.insert(0, {
                "role": "system",
                "content": f"Previous relevant information:\n{memory_context}",
                "timestamp": datetime.now().isoformat()
            })

        # Prepare messages for the model
        messages = filtered_history + [{"role": "user", "content": req.prompt}]

        try:
            # Get response from model
            response = await self.model_selector.query_ollama(messages, model, self.http_client)

            # --- Append reference to original book if RAG was used ---
            if rag_doc and rag_meta:
                ref_title = rag_meta.get("book_title", "original source")
                ref_info = rag_meta.get("book_info", "")
                response += f"\n\n<b>Reference:</b> {ref_title}"
                if ref_info:
                    response += f" ({ref_info})"

            # Format the response with HTML
            formatted_response = self.format_response(response)

            # Store response in history
            filtered_history.append({
                "role": "assistant",
                "content": response,  # Store original unformatted response
                "timestamp": datetime.now().isoformat()
            })
            await self.conv_service.save_history(conv_id, filtered_history)

            # Store new memories in background
            if req.use_memory:
                background_tasks.add_task(
                    self.memory_service.store_memory,
                    req.prompt,
                    conv_id,
                    language
                )
                background_tasks.add_task(
                    self.memory_service.store_memory,
                    response,
                    conv_id,
                    language
                )

            # Calculate processing time
            processing_time = time.time() - start_time

            return LLMResponse(
                response=formatted_response,  # Return formatted response
                conversation_id=conv_id,
                category=category,
                model_used=model,
                processing_time=processing_time,
                memories_used=memories_used if memories_used else None
            )

        except Exception as e:
            logger.error(f"Error processing chat: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_filtered_history(self, conv_id: str, query_language: str) -> List[Dict[str, str]]:
        """Helper method to get language-filtered conversation history"""
        full_history = await self.conv_service.get_history(conv_id)
        return [
            msg for msg in full_history
            if msg.get("role") == "system" or
            self.model_selector.language_detector.detect(msg.get("content", "")) == query_language
        ]

# ------------------------------------------------------------
# Services: ChromaDB Document Service (SRP)
# ------------------------------------------------------------
class ChromaDBDocumentService:
    def __init__(self, embedding_service: SentenceTransformerEmbeddingService):
        self.embedding_service = embedding_service
        self.chroma_client = chromadb.Client()
        
        # Initialize collections
        self.godic_collection = self.chroma_client.get_or_create_collection(
            name="chroma_data_godic",
            embedding_function=None  # We'll use our own embedding service
        )
        self.english_collection = self.chroma_client.get_or_create_collection(
            name="english_collection",
            embedding_function=None
        )
        self.arabic_collection = self.chroma_client.get_or_create_collection(
            name="arabic_collection",
            embedding_function=None
        )
        
        logger.info("ChromaDB collections initialized")

    async def query_documents(self, query: str, language: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Query documents from the godic collection based on query embedding"""
        try:
            # Normalize and get embedding for the query
            normalized_query = normalize_arabic(query) if language.lower() == "arabic" else query
            query_embedding = await self.embedding_service.get_embedding(normalized_query, language=language)
            
            # Query the collection
            results = self.godic_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit
            )
            
            # Format results
            documents = []
            if results and "documents" in results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    documents.append({
                        "content": doc,
                        "metadata": results["metadatas"][0][i] if "metadatas" in results and results["metadatas"] else {},
                        "id": results["ids"][0][i] if "ids" in results and results["ids"] else f"doc_{i}"
                    })
            
            return documents
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return []

    async def store_interaction(self, user_message: str, assistant_response: str, 
                               language: str, conversation_id: str) -> None:
        """Store user-assistant interaction in the appropriate collection"""
        try:
            # Determine which collection to use based on language
            collection = self.arabic_collection if language.lower() == "arabic" else self.english_collection
            
            # Generate unique IDs for the messages
            user_id = f"user_{conversation_id}_{uuid.uuid4()}"
            assistant_id = f"assistant_{conversation_id}_{uuid.uuid4()}"
            
            # Get embeddings
            user_embedding = await self.embedding_service.get_embedding(user_message, language=language)
            assistant_embedding = await self.embedding_service.get_embedding(assistant_response, language=language)
            
            # Add to collection
            collection.add(
                ids=[user_id, assistant_id],
                embeddings=[user_embedding, assistant_embedding],
                documents=[user_message, assistant_response],
                metadatas=[
                    {"type": "user", "conversation_id": conversation_id, "timestamp": datetime.now().isoformat()},
                    {"type": "assistant", "conversation_id": conversation_id, "timestamp": datetime.now().isoformat()}
                ]
            )
            
            logger.info(f"Stored interaction in {language} collection for conversation {conversation_id}")
        except Exception as e:
            logger.error(f"Error storing interaction in ChromaDB: {e}")

# ------------------------------------------------------------
# FastAPI Application and Dependency Injection
# ------------------------------------------------------------

app = FastAPI(
    version="3.0.0",
    default_response_class=JSONResponse,
)

origins = [
    "null",  # For local file access
    "file://",
    "http://localhost",
    "http://localhost:5000",
    "http://localhost:8080",
    "http://127.0.0.1",
    "http://127.0.0.1:5000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:5500",
    "https://thrush-hot-radically.ngrok-free.app",
    "*"  # Allow all origins temporarily for debugging
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    redis_client = await aioredis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
    
    # Use the SentenceTransformerEmbeddingService instead of the Ollama-based one
    embedding_service = SentenceTransformerEmbeddingService(settings, http_client)
    
    # Initialize Redis services
    conv_repo = RedisConversationRepository(redis_client, settings)
    conv_service = ConversationService(conv_repo)
    memory_service: IMemoryService = RedisMemoryService(redis_client, embedding_service)
    session_manager = RedisSessionManager(redis_client, settings)
    
    # Initialize ChromaDB service
    chroma_service = ChromaDBDocumentService(embedding_service)
    
    model_selector = ModelSelector(settings)
    chat_controller = ChatController(conv_service, memory_service, http_client, model_selector)
    supabase_repo = SupabaseRepository(settings)
    
    # Store services in app state
    app.state.chat_controller = chat_controller
    app.state.supabase_repo = supabase_repo
    app.state.session_manager = session_manager
    app.state.chroma_service = chroma_service

    cleanup_task = asyncio.create_task(periodic_cleanup(app))
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    await http_client.aclose()
    await redis_client.close()

app.router.lifespan_context = lifespan

@app.get("/api/chats/{chat_id}/conversations", response_model=List[ConversationRecord])
async def get_conversation_history(chat_id: int, request: Request):
    """
    Retrieve the list of messages for the given chat_id,
    in chronological order.
    """
    repo: SupabaseRepository = request.app.state.supabase_repo
    rows = await repo.get_conversation_history(chat_id)
    # rows is a list of dicts with keys matching ConversationRecord
    return [ConversationRecord(**row) for row in rows]

@app.get("/api/chats/thread/{thread_id}")
async def get_chat_by_thread_id(thread_id: str, request: Request):
    """Get chat details by thread_id"""
    repo: SupabaseRepository = request.app.state.supabase_repo
    chat = await repo.get_chat_by_thread_id(thread_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return ChatRecord(**chat)

@app.get("/api/user-chats")
async def get_user_chats(request: Request, authorization: str = Header(..., alias="Authorization")):
    """Get all chats for the authenticated user."""
    try:
        repo: SupabaseRepository = request.app.state.supabase_repo

        # Parse & validate token
        try:
            scheme, token = authorization.split()
            if scheme.lower() != "bearer":
                raise HTTPException(401, "Invalid authorization scheme")
        except ValueError:
            raise HTTPException(401, "Invalid authorization format")

        # Get user from token
        auth_resp = repo.supabase.auth.get_user(token)
        user = getattr(auth_resp, "user", None) or auth_resp.data.get("user")
        if not user or not getattr(user, "id", None):
            raise HTTPException(401, "Invalid user token")

        # Get user's chats
        chats = await repo.get_user_chats(user.id)
        return chats

    except Exception as e:
        logger.error(f"Error getting user chats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=LLMResponse)
async def chat(req: PromptRequest, background_tasks: BackgroundTasks, request: Request):
    controller: ChatController = request.app.state.chat_controller
    return await controller.process_chat(req, background_tasks)

@app.post("/api/chats", response_model=ChatRecord)
async def create_chat(
    chat: ChatRecord,
    request: Request,
    authorization: str = Header(..., alias="Authorization")
):
    """Create a placeholder chat with a default title."""
    try:
        repo: SupabaseRepository = request.app.state.supabase_repo

        # Parse & validate token
        try:
            scheme, token = authorization.split()
            if scheme.lower() != "bearer":
                raise HTTPException(401, "Invalid authorization scheme")
        except ValueError:
            raise HTTPException(401, "Invalid authorization format")

        # Validate user
        try:
            auth_resp = repo.supabase.auth.get_user(token)
            user = getattr(auth_resp, "user", None) or auth_resp.data.get("user")
            if not user or not getattr(user, "id", None):
                raise HTTPException(401, "Invalid user token")
        except Exception as e:
            logger.error(f"Auth error: {str(e)}")
            raise HTTPException(401, "Authentication failed")

        # Create placeholder chat
        chat.user_id = user.id
        chat.title = "New Chat"  # Default placeholder title
        result = await repo.create_chat(chat)
        if not result:
            raise HTTPException(500, "Failed to create chat")
        return ChatRecord(**result)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(500, f"Internal server error: {str(e)}")


@app.post("/api/chat-with-db")
async def chat_with_db(
    req: PromptRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    authorization: Optional[str] = Header(None, alias="Authorization")
):
    """Chat endpoint that also stores the conversation in the database."""
    controller: ChatController = request.app.state.chat_controller
    repo = request.app.state.supabase_repo
    session_manager = request.app.state.session_manager
    chroma_service = request.app.state.chroma_service

    user_id = None
    reference_id = None

    # Extract user_id from token if provided
    if authorization:
        try:
            scheme, token = authorization.split()
            if scheme.lower() == "bearer":
                auth_resp = repo.supabase.auth.get_user(token)
                user = getattr(auth_resp, "user", None) or auth_resp.data.get("user")
                if user and getattr(user, "id", None):
                    user_id = user.id
                    
                    # Get or initialize user session
                    user_session = await session_manager.get_session(user_id)
                    if not user_session:
                        user_session = {"last_access": datetime.now().isoformat()}
                        await session_manager.set_session(user_id, user_session)
        except Exception as e:
            logger.warning(f"Auth token processing error: {e}")

    # Try to get cached response for identical queries (simple caching)
    cache_key = f"query:{hash(req.prompt)}"
    cached_response = await session_manager.cache_get(cache_key)
    if cached_response and not req.conversation_id:  # Only use cache for new conversations
        logger.info(f"Using cached response for query: {req.prompt[:30]}...")
        return cached_response

    # Query ChromaDB before processing chat to get document ID
    try:
        query_emb = get_query_embeddings(req.prompt)
        document, metadata = chroma_service.retrieve_documents(query_emb, n_results=1)
        if metadata and 'doc_id' in metadata:
            reference_id = metadata['doc_id']
            logger.debug(f"Retrieved ChromaDB document ID: {reference_id}")
    except Exception as e:
        logger.warning(f"Error retrieving ChromaDB document ID: {e}")
        reference_id = None

    # Process the chat request
    response = await controller.process_chat(req, background_tasks)
    
    # Cache the response for future use (5 minute TTL)
    if not req.conversation_id:  # Only cache responses for new conversations
        await session_manager.cache_set(cache_key, response, ttl=300)
    
    # Store interaction in ChromaDB for future retrieval
    try:
        language = controller.model_selector.language_detector.detect(req.prompt)
        await chroma_service.store_interaction(
            user_message=req.prompt,
            assistant_response=response.response,
            language=language,
            conversation_id=response.conversation_id
        )
    except Exception as e:
        logger.warning(f"Error storing interaction in ChromaDB: {e}")

    # Store in database if conversation_id is provided
    if req.conversation_id and not req.stream:
        # Get chat by thread_id
        chat = await repo.get_chat_by_thread_id(req.conversation_id)
        if chat:
            chat_id = chat.get("id")

            if chat_id:
                # Store the user message
                await repo.add_conversation_message(ConversationRecord(
                    chat_id=chat_id,
                    type="user",
                    content=req.prompt
                ))

                # Store the assistant response with processing time, reference, and category
                await repo.add_conversation_message(ConversationRecord(
                    chat_id=chat_id,
                    type="assistant",
                    content=response.response,
                    time=response.processing_time,
                    reference=reference_id,  # Store ChromaDB document ID
                    category=response.category  # Store the category used for model selection
                ))

                # Update title if it's the first message
                if chat.get("title") == "New Chat":
                    updated_title = req.prompt[:50] + "..." if len(req.prompt) > 50 else req.prompt
                    result = repo.supabase.table('chat').update(
                        {"title": updated_title}
                    ).eq("id", chat_id).execute()
                    if result.data:
                        chat["title"] = result.data[0]["title"]  # Ensure the title matches Supabase
        else:
            # Create new chat using the first prompt as the title
            new_chat = ChatRecord(
                thread_id=req.conversation_id,
                title=req.prompt[:50] + "..." if len(req.prompt) > 50 else req.prompt,
                user_id=user_id
            )
            chat_result = await repo.create_chat(new_chat)
            if chat_result:
                chat_id = chat_result.get("id")
                if chat_id:
                    await repo.add_conversation_message(ConversationRecord(
                        chat_id=chat_id,
                        type="user",
                        content=req.prompt
                    ))
                    await repo.add_conversation_message(ConversationRecord(
                        chat_id=chat_id,
                        type="assistant",
                        content=response.response,
                        time=response.processing_time,
                        reference=reference_id,  # Include the ChromaDB document ID
                        category=response.category  # Include the category used for model selection
                    ))
    return {"response": response, "updated_title": chat.get("title"), "processing_time": response.processing_time, "category": response.category}

@app.put("/api/chats/{chat_id}")
async def update_chat_title(
    chat_id: int,
    data: Dict[str, str],
    request: Request,
    authorization: str = Header(..., alias="Authorization")
):
    """Update the title of a chat in Supabase only if it's the first message."""
    repo: SupabaseRepository = request.app.state.supabase_repo
    try:
        # Parse & validate token
        try:
            scheme, token = authorization.split()
            if scheme.lower() != "bearer":
                raise HTTPException(401, "Invalid authorization scheme")
        except ValueError:
            raise HTTPException(401, "Invalid authorization format")

        # Check if the chat already has messages
        existing_messages = await repo.get_conversation_history(chat_id)
        if existing_messages:
            logger.debug(f"Chat {chat_id} already has messages. Title update skipped.")
            raise HTTPException(status_code=400, detail="Chat title can only be updated for the first message.")

        # Update title in Supabase
        logger.debug(f"Updating title for chat {chat_id} to '{data.get('title')}'")
        result = repo.supabase.table('chat').update(
            {"title": data.get("title")}
        ).eq("id", chat_id).execute()

        if not result.data or len(result.data) == 0:
            logger.error(f"Failed to update title for chat {chat_id} in Supabase.")
            raise HTTPException(status_code=404, detail="Chat not found or update failed")

        # Return the updated chat data
        logger.debug(f"Title for chat {chat_id} updated successfully in Supabase.")
        return {"message": "Chat title updated successfully", "data": result.data[0]}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error updating chat title: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def periodic_cleanup(app: FastAPI):
    while True:
        try:
            logger.info("Running cleanup task")
            # Implement cleanup logic here if necessary
        except Exception as e:
            logger.error("Cleanup error: %s", e)
        await asyncio.sleep(get_settings().CLEANUP_INTERVAL)

# ------------------------------------------------------------
# Application Entry Point with Static Ngrok Domain Integration
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = 5005

    use_ngrok = os.getenv("USE_NGROK", "false")

    if use_ngrok:
        ngrok.set_auth_token(os.getenv("NGROK_AUTHTOKEN", "2vfcy1ncjgggNx5Ru2qDnJY3SmD_4ef6eJd5Zm554fJa8vrgk"))
        tunnel = ngrok.connect(
            addr=port,
            hostname="thrush-hot-radically.ngrok-free.app",  # Updated Ngrok URL
            proto="http",
        )
        logger.info(f"Ngrok tunnel established at: {tunnel.public_url}")

    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)