"""
Configuration module for the ARA assistant.

Responsibilities:
- Load environment variables from .env via python-dotenv.
- Expose API keys for Groq, HuggingFace, Serper, NewsAPI, and Tavily.
- Configure paths for SQLite checkpoint database and ChromaDB vector store.
- Initialize LangSmith tracing if enabled.
- Create required directories at import time.

Side Effects:
- Calls load_dotenv() on import.
- Sets os.environ keys for LangChain tracing if LANGSMITH_TRACING=true.
- Creates data/sqlite/ and data/chroma/ directories if they don't exist.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# --- API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_API_KEY_BACKUP = os.getenv("GROQ_API_KEY_BACKUP", "").strip()

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "").strip()

# --- RAG Storage
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/sqlite/ara.db")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

# --- LangSmith
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "").strip()
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "ARA")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")


def setup_langsmith():
    """
    Enable LangSmith tracing by setting LangChain environment variables.

    Checks if LANGSMITH_TRACING is true and LANGSMITH_API_KEY is provided.
    If both conditions are met, sets LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY,
    LANGCHAIN_PROJECT, and LANGCHAIN_ENDPOINT in os.environ.

    Returns:
        bool: True if tracing was enabled, False otherwise.

    Side Effects:
        Modifies os.environ with LangChain tracing configuration.
    """
    if LANGSMITH_TRACING and LANGSMITH_API_KEY:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
        os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT
        return True
    return False


def ensure_directories():
    """
    Create parent directories for SQLite database and ChromaDB persistence.

    Creates:
        - Parent directory of SQLITE_DB_PATH (e.g., ./data/sqlite/)
        - CHROMA_PERSIST_DIR (e.g., ./data/chroma/)

    Side Effects:
        Creates directories on filesystem if they don't exist.
    """
    Path(SQLITE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)


langsmith_enabled = setup_langsmith()
ensure_directories()
