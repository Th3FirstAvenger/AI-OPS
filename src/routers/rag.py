"""RAG Router for document management and collection operations"""
import logging
from typing import List, Optional
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends
from pathlib import Path
import tempfile
import json


from src.core.knowledge import DocumentProcessor, MaintenanceUtils, Topic
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

    """Upload a document to a collection."""
    if not store:
        return {"error": "RAG is not available"}
        
    if collection_name not in store.collections:
        return {"error": f"Collection '{collection_name}' does not exist"}
        
    # Process topics
    topic_list = topics.split(',') if topics and topics.strip() else []
    
    # Create necessary directories
    documents_dir = Path.home() / '.aiops' / 'knowledge' / 'documents' / collection_name
    database_dir = Path.home() / '.aiops' / 'database'
    documents_dir.mkdir(parents=True, exist_ok=True)
    database_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary file for processing
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
    temp_path = temp_file.name
    
    try:
        # Read file content
        content = await file.read()
        
        # Save original file to documents directory
        document_path = documents_dir / file.filename
        with open(document_path, 'wb') as f:
            f.write(content)
        
        # Write to temp file for processing
        with open(temp_path, 'wb') as f:
            f.write(content)
        
        # Process document
        document = DocumentProcessor.from_file(temp_path, topic_list)
        
        # Upload to collection
        store.upload(document, collection_name)
                # Add topics to collection
        
        for topic in topic_list:
            # Create Topic object if needed
            topic_obj = Topic(topic)
            # Add to collection's topics if not already there
            if topic_obj not in store.collections[collection_name].topics:
                store.collections[collection_name].topics.append(topic_obj)
        
        # Update the collection metadata file
        if hasattr(store, 'save_metadata') and callable(store.save_metadata):
            store.save_metadata(store.collections[collection_name])
        
        # Update markdown index
        index_path = database_dir / "markdown_upload_files.json"
        
        # Load existing index or create new
        if index_path.exists():
            with open(index_path, 'r') as f:
                try:
                    index_data = json.load(f)
                except:
                    index_data = []
        else:
            index_data = []
        
        # Create entry for this document
        document_entry = {
            "title": file.filename,
            "topics": topic_list,
            "path": str(document_path),
            "collection": collection_name
        }
        
        # Update or add entry
        updated = False
        for i, entry in enumerate(index_data):
            if entry.get("title") == file.filename and entry.get("collection") == collection_name:
                index_data[i] = document_entry
                updated = True
                break
        
        if not updated:
            index_data.append(document_entry)
        
        # Save updated index
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        return {
            "success": True,
            "message": f"Document '{file.filename}' uploaded with topics: {', '.join(topic_list)}"
        }
    except Exception as e:
        logger.error(f"Error uploading document to '{collection_name}': {e}")
        import traceback
        traceback.print_exc()
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