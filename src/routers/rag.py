"""
RAG router for AI-OPS API.
"""
import json
from typing import Optional, Annotated
from functools import lru_cache

from fastapi import APIRouter, File, Form, UploadFile, Depends, HTTPException
from pathlib import Path

from src.config import RAG_SETTINGS
from src.core.knowledge import Collection, Topic, QdrantStore
from src.core.knowledge.collections import MarkdownParser, Document
from src.utils import get_logger

logger = get_logger(__name__)

rag_router = APIRouter()

@lru_cache()
def get_rag_store() -> QdrantStore:
    """
    Singleton factory para crear y reutilizar la instancia de Store.
    El decorador lru_cache asegura que la función solo se ejecute una vez.
    """
    try:
        logger.info("Initializing RAG store...")
        store = QdrantStore(
            base_path=str(Path(Path.home() / '.aiops')),
            url=RAG_SETTINGS.RAG_URL,
            embedding_url=RAG_SETTINGS.EMBEDDING_URL,
            embedding_model=RAG_SETTINGS.EMBEDDING_MODEL,
            in_memory=RAG_SETTINGS.IN_MEMORY
        )
        logger.info("RAG store initialized successfully")
        return store
    except Exception as e:
        logger.error(f"Failed to initialize RAG store: {e}")
        # Aquí podrías devolver una versión simulada de Store para desarrollo
        # o simplemente dejar que se propague la excepción
        raise

@rag_router.get('/collections/list')
def list_collections(store: QdrantStore = Depends(get_rag_store)):
    """
    Returns available Collections.
    Returns a JSON list of available Collections.
    """
    try:
        # Primero intentamos obtener las colecciones del store
        store_collections = [c.to_dict() for c in store.collections.values()]
        
        if store_collections:
            return store_collections
        
        # Si no hay colecciones en el store, consultamos directamente a Qdrant
        try:
            import httpx
            qdrant_response = httpx.get(f"{RAG_SETTINGS.RAG_URL}/collections")
            qdrant_response.raise_for_status()
            qdrant_data = qdrant_response.json()
            
            if 'result' in qdrant_data and 'collections' in qdrant_data['result']:
                collections = []
                for collection in qdrant_data['result']['collections']:
                    # Creamos una estructura compatible con lo que espera el cliente
                    collections.append({
                        'title': collection['name'],
                        'topics': ['Unknown'],  # No podemos saber los temas sin metadata
                        'documents': []  # No podemos saber los documentos sin metadata
                    })
                return collections
        except Exception as qdrant_err:
            logger.warning(f"Failed to fetch collections from Qdrant: {qdrant_err}")
        
        # Si todo falla, devolvemos lista vacía
        return []
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")
    

@rag_router.post('/collections/new')
async def create_collection(
    title: str = Form(...),
    file: Optional[UploadFile] = File(None),
    store: QdrantStore = Depends(get_rag_store)
):
    """
    Creates a new Collection.
    :param file: uploaded file
    :param title: unique collection title
    :param store: RAG store dependency

    Returns error message for any validation error.
    1. title should be unique
    2. the file should follow one of the formats:
       - JSON: A list of documents in the following format:
         [
             {
                 "title": "document title",
                 "content": "...",
                 "category": "document topic"
             },
             ...
         ]
       - Markdown: A valid markdown file.
    """
    try:
        if title in list(store.collections.keys()):
            return {'error': f'A collection named "{title}" already exists.'}

        if not file:
            available_collections = list(store.collections.values())
            last_id = available_collections[-1].collection_id if available_collections else 0
            store.create_collection(
                Collection(
                    collection_id=last_id + 1,
                    title=title,
                    documents=[],
                    topics=[]
                )
            )
        else:
            if file.filename.endswith('.md'):
                # Process Markdown file
                contents = await file.read()
                markdown_parser = MarkdownParser()
                document_content = contents.decode('utf-8')
                # Parse markdown into document chunks (if you want to use chunks later)
                md_chunks = markdown_parser.parse_markdown(document_content, file.filename)
                
                # For this example, we create a single Document using the entire markdown content.
                document = Document(
                    name=file.filename,
                    content=document_content,
                    topic=Topic("Markdown")  # Set a default or extract topic as needed
                )
                
                available_collections = list(store.collections.values())
                last_id = available_collections[-1].collection_id if available_collections else 0
                new_collection = Collection(
                    collection_id=last_id + 1,
                    title=title,
                    documents=[document],
                    topics=[Topic("Markdown")]
                )
                try:
                    store.create_collection(new_collection)
                except RuntimeError as create_err:
                    return {'error': str(create_err)}
            
            elif file.filename.endswith('.json'):
                # Process JSON file
                contents = await file.read()
                try:
                    collection_data: list[dict] = json.loads(contents.decode('utf-8'))
                except (json.decoder.JSONDecodeError, UnicodeDecodeError):
                    return {'error': 'Invalid file'}
                
                try:
                    new_collection = Collection.from_dict(title, collection_data)
                except ValueError as schema_err:
                    return {'error': schema_err}
                
                try:
                    store.create_collection(new_collection)
                except RuntimeError as create_err:
                    return {'error': str(create_err)}
            else:
                return {'error': 'Invalid file. Only JSON or Markdown files are allowed.'}

        return {'success': f'{title} created successfully.'}
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        return {'error': f'An unexpected error occurred: {str(e)}'}


@rag_router.post('/collections/upload')
async def upload_document(store: QdrantStore = Depends(get_rag_store)):
    """Uploads a document to an existing collection."""
    # TODO: Implementar subida de documentos a colecciones existentes
    return {"message": "Not implemented yet"}


# Endpoint para búsqueda en colecciones
@rag_router.get('/collections/{collection_name}/search')
async def search_collection(
    collection_name: str,
    query: str,
    limit: int = 5,
    threshold: float = 0.5,
    store: QdrantStore = Depends(get_rag_store)
):
    """
    Search in a specific collection.
    """
    try:
        if collection_name not in store.collections:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
        
        results = store.retrieve_from(
            query=query,
            collection_name=collection_name,
            limit=limit,
            threshold=threshold
        )
        
        if not results:
            return {"results": [], "message": "No relevant documents found"}
        
        return {"results": results}
    except Exception as e:
        logger.error(f"Error searching collection: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching collection: {str(e)}")