"""Tool package"""

from tool_parse import ToolRegistry

# from src.core.tools.exploit_db import ExploitDB
from src.core.tools.web_search import Search
# from src.core.tools.terminal import Terminal
from src.utils import get_logger
from src.core.knowledge.store import QdrantStore
from src.config import RAG_SETTINGS

logger = get_logger(__name__)


TOOL_REGISTRY = ToolRegistry()
# Search tool dont implemented yet
"""SEARCH = Search()


@TOOL_REGISTRY.register(description=SEARCH.usage)
def search_web(search_query: str):
    return SEARCH.run(search_query)"""

"""
This implementation adds a new tool that allows the agent
to search document collections using RAG.
"""


class RAGSearch:
    """Tool for searching in document collections using RAG"""
    
    usage = "Search for information in document collections using RAG (Retrieval-Augmented Generation)"
    
    def __init__(self):
        """Initialize the RAG search tool"""
        try:
            self.rag_store = QdrantStore()
            logger.info("RAGSearch: Tool successfully initialized")
        except Exception as e:
            logger.error(f"RAGSearch: Error initializing tool: {str(e)}")
            self.rag_store = None
    
    def run(self, search_query: str, collection: str = None, limit: int = 3, threshold: float = 0.5) -> str:
        """
        Search for information in document collections.
        """
        logger.info(f"RAGSearch: Running search for query: '{search_query}', collection: '{collection}'")
        
        if not self.rag_store:
            logger.error("RAGSearch: rag_store not initialized")
            return "Error: The RAG search tool is not properly initialized."
        
        try:
            # Get available collections
            available_collections = list(self.rag_store.collections.keys())
            logger.info(f"RAGSearch: Available collections: {available_collections}")
            
            if not available_collections:
                logger.warning("RAGSearch: No collections available")
                return "No collections available to search."
            
            # If no collection specified, search in all
            collections_to_search = [collection] if collection and collection in available_collections else available_collections
            logger.info(f"RAGSearch: Searching in collections: {collections_to_search}")
            
            all_results = []
            
            # Search in each collection
            for coll_name in collections_to_search:
                try:
                    logger.info(f"RAGSearch: Retrieving from collection '{coll_name}'")
                    results = self.rag_store.retrieve_from(
                        query=search_query,
                        collection_name=coll_name,
                        limit=limit,
                        threshold=threshold
                    )
                    
                    if results:
                        logger.info(f"RAGSearch: Found {len(results)} results in '{coll_name}'")
                        # Format results with collection name
                        for result in results:
                            all_results.append({
                                "collection": coll_name,
                                "text": result
                            })
                    else:
                        logger.info(f"RAGSearch: No results found in '{coll_name}'")
                except Exception as coll_error:
                    logger.warning(f"RAGSearch: Error searching collection {coll_name}: {str(coll_error)}")
            
            # Format response
            if not all_results:
                logger.warning(f"RAGSearch: No relevant information found for query: '{search_query}'")
                return f"No relevant information found for: '{search_query}'"
            
            formatted_results = "\n\n".join([
                f"[Collection: {result['collection']}]\n{result['text']}"
                for result in all_results[:limit]
            ])
            
            logger.info(f"RAGSearch: Returning {len(all_results[:limit])} results")
            return f"Results for '{search_query}':\n\n{formatted_results}"
            
        except Exception as e:
            logger.error(f"Error in RAGSearch.run: {str(e)}")
            return f"Error performing search: {str(e)}"

# Initialize RAGSearch instance with verbose logging
RAG_SEARCH = RAGSearch()

# Register the RAG search tool
@TOOL_REGISTRY.register(description=RAG_SEARCH.usage)
def search_rag(search_query: str, collection: str = None):
    """
    Search for information in document collections.
    
    Args:
        search_query: The search query
        collection: Specific collection name (optional)
    """
    logger.info(f"Tool called: search_rag('{search_query}', '{collection}')")
    result = RAG_SEARCH.run(search_query, collection)
    logger.info(f"Tool result length: {len(result) if result else 0} characters")
    return result