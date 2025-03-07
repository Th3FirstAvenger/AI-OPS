"""Tool package"""

from tool_parse import ToolRegistry

# from src.core.tools.exploit_db import ExploitDB
from src.core.tools.web_search import Search
from src.core.tools.rag_search import RAGSearch
# from src.core.tools.terminal import Terminal
from src.config import AGENT_SETTINGS

TOOL_REGISTRY = ToolRegistry()
SEARCH = Search()
RAG_SEARCH = RAGSearch()

import logging
logger = logging.getLogger(__name__)


@TOOL_REGISTRY.register(description=SEARCH.usage)
def search_web(search_query: str):
    """Make an online search using a query string."""
    return SEARCH.run(search_query)

if AGENT_SETTINGS.USE_RAG:
    logger.info("Registering RAG search tool...")
    @TOOL_REGISTRY.register(description=RAG_SEARCH.usage)
    def search_rag(
        rag_query: str, 
        collection: str = None, 
        topics: str = None, 
        collection_title: str = None,
        detail_level: str = "medium"
    ) -> str:
        """
        Search the knowledge base with advanced options.
        
        Args:
            rag_query: The search query
            collection: Optional specific collection name to search in
            topics: Optional comma-separated list of topics to filter by
            collection_title: Optional collection title pattern to match
            detail_level: Amount of detail to return ("brief", "medium", or "detailed")
            
        Returns:
            Retrieved information from the knowledge base
        """
        logger.info(f"Searching RAG with query: {rag_query}")
        return RAG_SEARCH.run(rag_query, collection, topics, collection_title, detail_level)