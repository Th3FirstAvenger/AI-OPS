# src/dependencies.py

"""
Initializes all the necessary dependencies for the API.
"""
import logging
from src.config import AGENT_SETTINGS, RAG_SETTINGS
from src.agent import Agent, build_agent
from pathlib import Path
from src.core.tools import TOOL_REGISTRY, RAG_SEARCH
from src.core.knowledge import load_rag

# Set up logging save path
log_path = Path.home() / '.aiops' / 'logs'
log_path.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=log_path / 'api.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)   
logger = logging.getLogger(__name__)

# Initialize the agent
logger.info("Initializing agent with model: %s", AGENT_SETTINGS.MODEL)
agent: Agent = build_agent(
    model=AGENT_SETTINGS.MODEL,
    inference_endpoint=AGENT_SETTINGS.ENDPOINT,
    provider=AGENT_SETTINGS.PROVIDER,
    provider_key=AGENT_SETTINGS.PROVIDER_KEY
)

# Initialize the enhanced store if RAG is enabled
logger.info("Initializing RAG system...")
store = None
if AGENT_SETTINGS.USE_RAG:
    logger.info("RAG is enabled. Initializing RAG system...")
    try:
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
        
        logging.info("RAG system initialized successfully. Available collections:")
        for cname, coll in store.collections.items():
            doc_topics = ", ".join([topic.name for topic in coll.topics])
            logging.info(f"- '{cname}': {doc_topics}")

        # Pass the store reference to the RAG_SEARCH tool
        RAG_SEARCH.set_store(store)
        
        # Log registered tools for debugging
        tool_names = [t["function"]["name"] for t in TOOL_REGISTRY.marshal('base')]
        logger.info("Registered tools after RAG initialization: %s", tool_names)
        
        # Log available collections
        if store:
            collection_names = list(store.collections.keys())
            logger.info("Available collections: %s", collection_names)
    except Exception as e:
        logger.error("Failed to initialize RAG system: %s", str(e))
        import traceback
        traceback.print_exc()
        # Set USE_RAG to False since initialization failed
        AGENT_SETTINGS.USE_RAG = False
else:
    logger.info("RAG is disabled in settings. Skipping RAG initialization.")

def get_agent():
    """Expose agent for Dependency Injection"""
    return agent

def get_store():
    """Expose store for Dependency Injection"""
    global store
    if store is None and AGENT_SETTINGS.USE_RAG:
        logger.error("RAG store is None but USE_RAG is True - attempting reinitialization")
        try:
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
            if store:
                RAG_SEARCH.set_store(store)
                logger.info("RAG store successfully reinitialized")
        except Exception as e:
            logger.error(f"RAG store reinitialization failed: {str(e)}")
    return store