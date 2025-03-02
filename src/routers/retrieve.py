"""
Retrieve endpoint for direct RAG queries
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional

from src.utils import get_logger
from src.routers.rag import get_rag_store
from src.core.knowledge.store import QdrantStore

logger = get_logger(__name__)

retrieve_router = APIRouter()

@retrieve_router.get("/retrieve")
async def retrieve(
    query: str, 
    collection: str, 
    limit: int = 5,
    threshold: float = 0.5,
    store: QdrantStore = Depends(get_rag_store)
) -> List[str]:
    """Retrieve documents from a collection using RAG"""
    try:
        logger.info(f"Retrieving from collection '{collection}' with query: {query}")
        
        # Buscar la colección de forma más flexible
        collection_found = False
        collection_key = None
        
        # Mostrar las colecciones disponibles para depuración
        available_collections = list(store.collections.keys())
        logger.info(f"Available collections: {available_collections}")
        
        # Primero intentar una coincidencia exacta
        if collection in store.collections:
            collection_found = True
            collection_key = collection
        else:
            # Intentar una coincidencia insensible a mayúsculas/minúsculas
            for key in store.collections.keys():
                if key.lower() == collection.lower():
                    collection_found = True
                    collection_key = key
                    break
        
        if not collection_found:
            logger.error(f"Collection '{collection}' not found. Available collections: {available_collections}")
            raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")
        
        # Realizar la búsqueda con la clave correcta
        logger.info(f"Using collection key: {collection_key}")
        results = store.retrieve_from(
            query=query,
            collection_name=collection_key,
            limit=limit,
            threshold=threshold
        )
        
        if not results:
            return ["No relevant documents found for the query."]
        
        return results
    except Exception as e:
        logger.error(f"Error retrieving from collection: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Error retrieving from collection: {str(e)}")