"""Knowledge Module: contains vector database and documents classes"""
from pathlib import Path
# Remove or comment out this import that's causing issues
# from tool_parse import ToolRegistry 
from src.core.knowledge.store import EnhancedStore
from src.core.knowledge.collections import Collection, Document, Topic
from src.core.knowledge.document_processor import DocumentProcessor, ChunkInfo
from src.core.knowledge.bm25_index import BM25Index, CollectionBM25Index
from src.core.knowledge.neural_reranker import NeuralReranker
from src.core.knowledge.batch_utils import batch_process
from src.core.knowledge.compression import compress_text, decompress_text
from src.core.knowledge.maintenance import MaintenanceUtils

from pathlib import Path
import json

import logging
logger = logging.getLogger(__name__)
def initialize_knowledge(vdb: EnhancedStore):
    """Used to initialize and keep updated the Knowledge Base.
    Already existing Collections will not be overwritten.
    :param vdb: the reference to the Knowledge Base"""
    knowledge_path = Path.home() / '.aiops' / 'knowledge'

    # Create directories if they don't exist
    knowledge_path.mkdir(parents=True, exist_ok=True)
    
    # Load collections from JSON files in the knowledge directory
    for dataset_file in knowledge_path.glob('*.json'):
        logger.info(f"Found JSON file: {dataset_file}")
        try:
            with open(dataset_file, 'r') as f:
                dataset_data = json.load(f)
            collection = Collection.from_dict(dataset_file.stem, dataset_data)
            if collection.title not in vdb.collections:
                vdb.create_collection(collection, progress_bar=True)
                save_collection_metadata(collection, knowledge_path)
                logger.info(f"[+] Created collection '{collection.title}'")
            else:
                logger.info(f"[+] Collection '{collection.title}' already exists")
        except Exception as e:
            logger.info(f"[!] Error loading dataset {dataset_file}: {e}")

    # Synchronize collections existing in Qdrant
    try:
        qdrant_collections = vdb._connection.get_collections().collections
        for qdrant_coll in qdrant_collections:
            coll_name = qdrant_coll.name
            
            # Skip if we already have this collection loaded
            if coll_name in vdb.collections:
                continue
                
            # Create a new collection object
            collection = Collection(
                collection_id=len(vdb._collections) + 1,
                title=coll_name,
                documents=[],
                topics=[],
                size=0
            )
            
            # Try to retrieve document information from Qdrant
            try:
                # Get a sample of points to extract document names and topics
                points = vdb._connection.scroll(
                    collection_name=coll_name,
                    limit=100,  # Get a reasonable sample
                    with_payload=True,
                    with_vectors=False
                )[0]
                
                # Extract unique document names and topics
                doc_names = set()
                all_topics = set()
                
                for point in points:
                    if 'doc_id' in point.payload:
                        doc_names.add(point.payload['doc_id'])
                    if 'topics' in point.payload and point.payload['topics']:
                        for topic in point.payload['topics']:
                            all_topics.add(topic)
                
                # Create placeholder documents
                for doc_name in doc_names:
                    collection.documents.append(Document(
                        name=doc_name,
                        content="",  # Empty content as placeholder
                        topics=[Topic(t) for t in all_topics],
                        source_type="text"
                    ))
                
                # Set topics for collection
                collection.topics = [Topic(t) for t in all_topics]
                
                # Set size based on Qdrant count
                count_result = vdb._connection.count(collection_name=coll_name)
                collection.size = count_result.count
                
                # Add to collections
                vdb._collections[coll_name] = collection
                logger.info(f"[+] Reconstructed collection '{coll_name}' from Qdrant with {len(doc_names)} documents")
                
                # Save metadata for future
                save_collection_metadata(collection, knowledge_path)
                
            except Exception as e:
                logger.error(f"Error reconstructing collection '{coll_name}' from Qdrant: {e}")
                
    except Exception as e:
        logger.error(f"[!] Error syncing Qdrant collections: {e}")
        
def save_collection_metadata(collection: Collection, knowledge_path: Path):
    """Guarda los metadatos de una colección en un archivo JSON."""
    metadata_file = knowledge_path / f"{collection.title}.json"
    with open(metadata_file, 'w') as f:
        json.dump(collection.to_dict(), f, indent=2)

def load_rag(
        rag_endpoint: str,
        in_memory: bool,
        embedding_model: str,
        embedding_url: str,
        tool_registry, 
        use_reranker: bool = True,
        reranker_provider: str = "ollama",
        reranker_model: str = "",
        reranker_confidence: float = 0.0
):
    """Initialize and load the RAG system"""
    logger.info(f"[+] Initializing RAG system with endpoint {rag_endpoint} and embedding model {embedding_model}")
    store = EnhancedStore(
        str(Path(Path.home() / '.aiops')),
        url=rag_endpoint,
        embedding_url=embedding_url,
        embedding_model=embedding_model,
        in_memory=in_memory,
        use_reranker=use_reranker,
        reranker_provider=reranker_provider,
        reranker_model=reranker_model,
        reranker_confidence=reranker_confidence
    )

    initialize_knowledge(store)
    available_documents = ''
    for cname, coll in store.collections.items():
        doc_topics = ", ".join([topic.name for topic in coll.topics])
        available_documents += f"- '{cname}': {doc_topics}\n"
    
    return store