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
    datasets_path = Path.home() / '.aiops' / 'datasets'
    knowledge_path = Path.home() / '.aiops' / 'knowledge'
    
    # Crear directorios si no existen
    datasets_path.mkdir(parents=True, exist_ok=True)
    knowledge_path.mkdir(parents=True, exist_ok=True)

    # Cargar datasets desde archivos JSON
    for dataset_file in datasets_path.glob('*.json'):
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

            # Procesar documentos Markdown
            if 'documents' in dataset_data:
                for doc_info in dataset_data['documents']:
                    if doc_info['path'].endswith('.md'):
                        try:
                            doc_path = datasets_path / doc_info['path']
                            document = DocumentProcessor.process_markdown_file(str(doc_path), doc_info.get('topics', []))
                            collection.documents.append(document)
                            logger.info(f"[+] Added document '{doc_path.name}' to '{collection.title}'")
                        except Exception as e:
                            logger.info(f"[!] Error processing Markdown file {doc_path}: {e}")
        except Exception as e:
            logger.info(f"[!] Error loading dataset {dataset_file}: {e}")

    # Sincronizar colecciones existentes en Qdrant
    try:
        qdrant_collections = vdb._connection.get_collections().collections
        for qdrant_coll in qdrant_collections:
            coll_name = qdrant_coll.name
            if coll_name not in vdb.collections:
                metadata_file = knowledge_path / f"{coll_name}.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    collection = Collection.from_dict(coll_name, metadata)
                    vdb._collections[coll_name] = collection
                    logger.info(f"[+] Loaded collection '{coll_name}' from metadata")
                else:
                    logger.info(f"[!] Collection '{coll_name}' exists in Qdrant but has no metadata file")
    except Exception as e:
        logger.info(f"[!] Error syncing Qdrant collections: {e}")
        
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