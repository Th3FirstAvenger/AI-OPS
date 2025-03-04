"""
Initializes all the necessary dependencies for the API.
"""
from src.config import AGENT_SETTINGS, RAG_SETTINGS
from src.agent import Agent, build_agent
from pathlib import Path
from src.core.tools import TOOL_REGISTRY
from src.core.knowledge import load_rag

# Initialize the agent
agent: Agent = build_agent(
    model=AGENT_SETTINGS.MODEL,
    inference_endpoint=AGENT_SETTINGS.ENDPOINT,
    provider=AGENT_SETTINGS.PROVIDER,
    provider_key=AGENT_SETTINGS.PROVIDER_KEY
)

# Initialize the enhanced store if RAG is enabled
store = None
if AGENT_SETTINGS.USE_RAG:
    store = load_rag(
        rag_endpoint=RAG_SETTINGS.RAG_URL,
        in_memory=RAG_SETTINGS.IN_MEMORY,
        embedding_model=RAG_SETTINGS.EMBEDDING_MODEL,
        embedding_url=RAG_SETTINGS.EMBEDDING_URL,
        tool_registry=TOOL_REGISTRY,
        use_reranker=RAG_SETTINGS.USE_RERANKER,
        reranker_provider=RAG_SETTINGS.RERANKER_PROVIDER,
        reranker_model=RAG_SETTINGS.RERANKER_MODEL,
        reranker_confidence=RAG_SETTINGS.RERANKER_CONFIDENCE
    )

def get_agent():
    """Expose agent for Dependency Injection"""
    return agent

def get_store():
    """Expose store for Dependency Injection"""
    return store