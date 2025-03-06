import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings


class AgentSettings(BaseSettings):
    """Setup for AI Agent"""
    MODEL: str = os.environ.get('MODEL', 'mistral')
    ENDPOINT: str = os.environ.get('ENDPOINT', 'http://localhost:11434')
    PROVIDER: str = os.environ.get('PROVIDER', 'ollama')
    PROVIDER_KEY: str = os.environ.get('PROVIDER_KEY', '')
    USE_RAG: bool = os.environ.get('USE_RAG', True)

class RAGSettings(BaseSettings):
    """Settings for Qdrant vector database"""
    RAG_URL: str = os.environ.get('RAG_URL', 'http://localhost:6333')
    IN_MEMORY: bool = os.environ.get('IN_MEMORY', True)
    EMBEDDING_MODEL: str = os.environ.get('EMBEDDING_MODEL', 'nomic-embed-text')
    EMBEDDING_URL: str = os.environ.get('ENDPOINT', 'http://localhost:11434')
    
    # Reranker settings
    USE_RERANKER: bool = os.environ.get('USE_RERANKER', True) 
    RERANKER_PROVIDER: str = os.environ.get('RERANKER_PROVIDER', 'ollama')
    RERANKER_MODEL: str = os.environ.get('RERANKER_MODEL', 'qllama/bge-reranker-large')  # Empty for default
    RERANKER_CONFIDENCE: float = float(os.environ.get('RERANKER_CONFIDENCE', '0.0'))


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
