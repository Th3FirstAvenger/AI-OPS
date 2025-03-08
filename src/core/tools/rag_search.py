"""RAG Search Tool Implementation with enhanced search capabilities"""
from typing import Optional, List, Dict
from pathlib import Path
import logging


logger = logging.getLogger(__name__)

class RAGSearch:
    """Implementation of RAG search functionality for the AI-OPS system"""
    
    name: str = 'RAG Search'
    usage: str = "Search documents in the RAG Vector Database using advanced hybrid retrieval with flexible collection targeting."
    
    def __init__(self):
        self.store = None
    
    def set_store(self, store):
        """Set the store reference after initialization"""
        self.store = store
        logger.info(f"RAG Search tool initialized with store: {self.store is not None}")
    
    def run(
        self, 
        rag_query: str, 
        collection: Optional[str] = None, 
        topics: Optional[str] = None, 
        collection_title: Optional[str] = None,
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
        logger.info(f"RAG Search called with query: {rag_query}")
        logger.info(f"Store available: {self.store is not None}")
        try:
            if not self.store:
                logger.error("RAG Search called but store is not initialized")
                return "Error: RAG system is not available."
            
            # Process parameters
            topic_list = [t.strip() for t in topics.split(',')] if topics else None
            
            # Determine limit based on detail level
            limit_map = {"brief": 1, "medium": 3, "detailed": 5}
            limit = limit_map.get(detail_level.lower(), 3)
            
            # Log search parameters
            logger.info(
                f"Executing RAG search: query='{rag_query}'"
            )
            
            # Use the new hybrid_search method for more flexible searching
            results_by_collection = self.store.hybrid_search(
                query=rag_query,
                collection_name=collection,
                collection_title=collection_title,
                topics=topic_list,
                limit=limit,
                rerank=True
            )
            
            if not results_by_collection:
                return "No relevant information found."
            
            # Format results
            formatted_results = []
            
            for coll_name, results in results_by_collection.items():
                # Add collection header
                collection_obj = self.store.collections[coll_name]
                formatted_results.append(f"## From collection: {collection_obj.title}")
                
                # Add results
                if detail_level.lower() == "brief":
                    # Just the top result from each collection
                    formatted_results.append(results[0])
                else:
                    # Multiple results
                    for i, result in enumerate(results):
                        if detail_level.lower() == "detailed" or i < limit:
                            formatted_results.append(result)
                
                # Add separator between collections
                formatted_results.append("-" * 60)
            
            return "\n\n".join(formatted_results)
            
        except Exception as e:
            error_msg = f"Error searching RAG system: {str(e)}"
            logger.error(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg