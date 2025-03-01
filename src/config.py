import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from typing import ClassVar


class AgentSettings(BaseSettings):
    """Setup for AI Agent"""
    MODEL: str = os.environ.get('MODEL', 'mistral')
    ENDPOINT: str = os.environ.get('ENDPOINT', 'http://localhost:11434')
    PROVIDER: str = os.environ.get('PROVIDER', 'ollama')
    PROVIDER_KEY: str = os.environ.get('PROVIDER_KEY', '')
    USE_RAG: bool = os.environ.get('USE_RAG', False)


class RAGSettings(BaseSettings):
    """Settings for Qdrant vector database"""
    RAG_URL: str = os.environ.get('RAG_URL', 'http://localhost:6333')
    RAG_API_KEY: str = os.environ.get('RAG_API_KEY', '')
    PROVIDER: str = os.environ.get('PROVIDER', 'ollama')
    EMBEDDING_MODEL: str = os.environ.get('EMBEDDING_MODEL', 'nomic-embed-text')
    USE_HYBRID: bool = os.environ.get('USE_HYBRID', 'True') == 'True'
    # Keep these for backward compatibility
    EMBEDDING_URL: str = os.environ.get('EMBEDDING_URL', 'http://localhost:11434')
    IN_MEMORY: bool = os.environ.get('IN_MEMORY', 'False') == 'True'
    RERANKER_MODEL: str = os.environ.get('RERANKER_MODEL', 'bge-reranker-large')

    DEFAULT_CHUNK_SIZE: ClassVar[int] = int(os.environ.get('DEFAULT_CHUNK_SIZE', 512))
    DEFAULT_CHUNK_OVERLAP: ClassVar[int] = int(os.environ.get('DEFAULT_CHUNK_OVERLAP', 128))
    DEFAULT_TOP_K: ClassVar[int] = int(os.environ.get('DEFAULT_TOP_K', 5))
    DEFAULT_RERANK_TOP_K: ClassVar[int] = int(os.environ.get('DEFAULT_RERANK_TOP_K', 5))


class APISettings(BaseSettings):
    """Setup for API"""
    ORIGINS: list = [
        # TODO
    ]
    PROFILE: bool = os.environ.get('PROFILE', False)


load_dotenv()
AGENT_SETTINGS = AgentSettings()
RAG_SETTINGS = RAGSettings()
API_SETTINGS = APISettings()
