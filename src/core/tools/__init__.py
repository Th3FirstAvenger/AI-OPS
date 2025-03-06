"""Tool package"""

from tool_parse import ToolRegistry

# from src.core.tools.exploit_db import ExploitDB
from src.core.tools.web_search import Search
from src.core.tools.rag_search import RAGSearch
# from src.core.tools.terminal import Terminal

TOOL_REGISTRY = ToolRegistry()
SEARCH = Search()
RAG_SEARCH = RAGSearch()

@TOOL_REGISTRY.register(description=SEARCH.usage)
def search_web(search_query: str):
    """Make an online search using a query string."""
    return SEARCH.run(search_query)

@TOOL_REGISTRY.register(description=RAG_SEARCH.usage)
def search_rag(rag_query: str, collection: str, topics: str = None, detail_level: str = "medium") -> str:
    """
    Search the knowledge base with advanced options.
    
    Args:
        rag_query: The search query
        collection: The collection name to search in
        topics: Optional comma-separated list of topics to filter by
        detail_level: Amount of detail to return ("brief", "medium", or "detailed")
        
    Returns:
        Retrieved information from the knowledge base
    """
    return RAG_SEARCH.run(rag_query, collection, topics, detail_level)