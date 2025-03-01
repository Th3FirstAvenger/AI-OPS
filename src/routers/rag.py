"""
Work In Progress, not mounted in API routes.
"""
import json
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile
from pathlib import Path

from src.config import RAG_SETTINGS
from src.core.knowledge import Collection, Store, Topic
from src.core.knowledge.collections import MarkdownParser, Document

rag_router = APIRouter()
# temporarily make store variable
store: Store | None = None
store = Store(
    base_path=str(Path(Path.home() / '.aiops')),
    url=RAG_SETTINGS.RAG_URL,
    embedding_url=RAG_SETTINGS.EMBEDDING_URL,
    embedding_model=RAG_SETTINGS.EMBEDDING_MODEL,
    in_memory=RAG_SETTINGS.IN_MEMORY
)


@rag_router.get('/collections/list')
def list_collections():
    """
    Returns available Collections.
    Returns a JSON list of available Collections.
    """
    if store:
        available_collections = [c.to_dict() for c in store.collections.values()]
        return available_collections
    else:
        return {}


@rag_router.post('/collections/new')
async def create_collection(
        title: str = Form(...),
        file: Optional[UploadFile] = File(None)
):
    """
    Creates a new Collection.
    :param file: uploaded file
    :param title: unique collection title

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
    if not store:
        return {'error': "RAG is not available"}
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
            # You could also decide to process md_chunks differently.
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
            # Process JSON file (as before)
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



@rag_router.post('/collections/upload')
async def upload_document():
    """Uploads a document to an existing collection."""
    # TODO: file vs ?
