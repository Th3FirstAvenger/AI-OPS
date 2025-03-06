"""RAG Search Tool Implementation"""
from typing import Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class RAGSearch:
    """Implementation of RAG search functionality for the AI-OPS system"""
    
    name: str = 'RAG Search'
    usage: str = "Search documents in the RAG Vector Database using advanced hybrid retrieval."
    
    def __init__(self):
        self.store = None
    
    def set_store(self, store):
        """Set the store reference after initialization"""
        self.store = store
        logger.info(f"RAG Search tool initialized with store: {self.store is not None}")
    
    def run(self, rag_query: str, collection: str, topics: Optional[str] = None, detail_level: str = "medium") -> str:
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
        try:
            if not self.store:
                logger.error("RAG Search called but store is not initialized")
                return "Error: RAG system is not available."
            
            # Check if collection exists
            if collection not in self.store.collections:
                available_collections = list(self.store.collections.keys())
                return f"Error: Collection '{collection}' not found. Available collections: {available_collections}"
            
            # Process parameters
            topic_list = [t.strip() for t in topics.split(',')] if topics else None
            
            # Determine limit based on detail level
            limit_map = {"brief": 1, "medium": 3, "detailed": 5}
            limit = limit_map.get(detail_level.lower(), 3)
            
            logger.info(f"Executing RAG search: query='{rag_query}', collection='{collection}', topics={topic_list}, limit={limit}")
            
            # Execute search with all enhancements
            results = self.store.hybrid_retrieve(
                query=rag_query,
                collection_name=collection,
                topics=topic_list,
                limit=limit,
                rerank=True
            )
            
            if not results:
                return "No relevant information found."
            
            # Format results based on detail level
            if detail_level.lower() == "brief":
                return results[0]
            
            # For medium and detailed, join with separators
            separator = "\n\n" + "-" * 40 + "\n\n"
            return separator.join(results)
            
        except Exception as e:
            error_msg = f"Error searching RAG system: {str(e)}"
            logger.error(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg