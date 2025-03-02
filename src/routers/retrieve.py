"""
Retrieve endpoint for direct RAG queries
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional

from src.utils import get_logger
from src.routers.rag import get_rag_store
from src.core.knowledge import Store

logger = get_logger(__name__)

retrieve_router = APIRouter()

@retrieve_router.get("/retrieve")
@retrieve_router.get("/retrieve")
async def retrieve(
    query: str, 
    collection: str, 
    limit: int = 5,
    threshold: float = 0.5,
    store: Store = Depends(get_rag_store)
) -> List[str]:
    """Retrieve documents from a collection using RAG"""
    try:
        logger.info(f"Retrieving from collection '{collection}' with query: {query}")
        
        # Modificar cómo accedemos a la colección
        try:
            # Generar embedding para la consulta
            embedding = store._encoder(store._embedding_model, query)['embedding']
            
            # Acceder directamente al cliente Qdrant
            hits = store._connection.search(
                collection_name=collection,  # Usar el nombre exactamente como viene
                query_vector=embedding,
                limit=limit,
                score_threshold=threshold
            )
            
            results = [points.payload.get('text', '') for points in hits if 'text' in points.payload]
            
            if results:
                return results
            else:
                return ["No relevant documents found for the query."]
                
        except Exception as direct_err:
            logger.error(f"Error with direct Qdrant access: {direct_err}")
            # Intentar el método original como fallback
        
        # Si el método directo falla, intentar el método original
        if collection not in store.collections:
            # Mostrar las colecciones para depuración
            available_collections = list(store.collections.keys())
            logger.error(f"Collection '{collection}' not found. Available: {available_collections}")
            raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")
        
        results = store.retrieve_from(
            query=query,
            collection_name=collection,
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