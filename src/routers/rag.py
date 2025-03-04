"""RAG Router for document management and collection operations"""
import logging
from typing import List, Optional
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends
from pathlib import Path
import tempfile

from src.core.knowledge import DocumentProcessor, MaintenanceUtils
from src.dependencies import get_store

logger = logging.getLogger(__name__)

rag_router = APIRouter(
    prefix="/collections",
    tags=["collections"],
    responses={404: {"description": "Not found"}}
)

@rag_router.get("/list")
async def list_collections(store = Depends(get_store)):
    """
    List all available collections in the knowledge base.
    
    Returns:
        List of collection metadata
    """
    try:
        if not store:
            return {"error": "RAG is not available"}
        
        collections = []
        for name, collection in store.collections.items():
            collections.append({
                "title": name,
                "topics": list(set([topic.name for topic in collection.topics])),
                "documents": [{"name": doc.name} for doc in collection.documents],
                "size": collection.size
            })
        
        return collections
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@rag_router.get("/{collection_name}")
async def get_collection(collection_name: str, store = Depends(get_store)):
    """
    Get detailed information about a specific collection.
    
    Args:
        collection_name: Name of the collection
        
    Returns:
        Collection details
    """
    try:
        if not store:
            return {"error": "RAG is not available"}
            
        collection = store.get_collection(collection_name)
        if not collection:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
            
        # Get integrity report
        integrity = MaintenanceUtils.verify_collection_integrity(store, collection_name)
            
        return {
            "title": collection.title,
            "topics": list(set([topic.name for topic in collection.topics])),
            "documents": [{"name": doc.name, "topics": [t.name for t in doc.topics]} for doc in collection.documents],
            "size": collection.size,
            "integrity": integrity
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting collection '{collection_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

@rag_router.post("/new")
async def create_collection(
    title: str = Form(...),
    file: Optional[UploadFile] = File(None),
    store = Depends(get_store)
):
    """
    Create a new collection.
    
    Args:
        title: Collection title
        file: Optional file containing collection data
        
    Returns:
        Success or error message
    """
    if not store:
        return {"error": "RAG is not available"}
        
    if title in store.collections:
        return {"error": f"Collection '{title}' already exists"}
        
    try:
        # Logic for creating collection from file or empty collection
        from src.core.knowledge import Collection
        collection = Collection(
            collection_id=1,  # Will be overridden by the store
            title=title,
            documents=[],
            topics=[]
        )
        store.create_collection(collection)
        return {"success": f"Collection '{title}' created successfully"}
    except Exception as e:
        logger.error(f"Error creating collection '{title}': {e}")
        return {"error": str(e)}

@rag_router.post("/{collection_name}/upload")
async def upload_document(
    collection_name: str,
    file: UploadFile = File(...),
    topics: str = Form(None),
    store = Depends(get_store)
):
    """
    Upload a document to a collection.
    
    Args:
        collection_name: Name of the collection
        file: Document file
        topics: Optional comma-separated list of topics
        
    Returns:
        Success or error message
    """
    if not store:
        return {"error": "RAG is not available"}
        
    if collection_name not in store.collections:
        return {"error": f"Collection '{collection_name}' does not exist"}
        
    # Process topics
    topic_list = topics.split(',') if topics and topics.strip() else []
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
    temp_path = temp_file.name
    
    try:
        # Write uploaded file to temp file
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Process document
        document = DocumentProcessor.from_file(temp_path, topic_list)
        
        # Upload to collection
        store.upload(document, collection_name)
        
        return {
            "success": True,
            "message": f"Document '{file.filename}' uploaded with topics: {', '.join(topic_list)}"
        }
    except Exception as e:
        logger.error(f"Error uploading document to '{collection_name}': {e}")
        return {"error": str(e)}
    finally:
        # Clean up temp file
        try:
            Path(temp_path).unlink()
        except:
            pass

@rag_router.post("/{collection_name}/maintenance")
async def maintain_collection(
    collection_name: str,
    operation: str = Form(...),
    store = Depends(get_store)
):
    """
    Perform maintenance operations on a collection.
    
    Args:
        collection_name: Name of the collection
        operation: Maintenance operation to perform
        
    Returns:
        Success or error message
    """
    if not store:
        return {"error": "RAG is not available"}
        
    if collection_name not in store.collections:
        return {"error": f"Collection '{collection_name}' does not exist"}
        
    try:
        if operation == "rebuild_bm25":
            success = MaintenanceUtils.rebuild_bm25_index(store, collection_name)
            if success:
                return {"success": f"BM25 index rebuilt for collection '{collection_name}'"}
            else:
                return {"error": "Failed to rebuild BM25 index"}
        elif operation == "verify":
            report = MaintenanceUtils.verify_collection_integrity(store, collection_name)
            return {"report": report}
        else:
            return {"error": f"Unknown operation: {operation}"}
    except Exception as e:
        logger.error(f"Error performing maintenance on '{collection_name}': {e}")
        return {"error": str(e)}