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
    knowledge_path.mkdir(parents=True, exist_ok=True)

    # Verificar conexión con Qdrant
    try:
        qdrant_collections = vdb._connection.get_collections().collections
        logger.info(f"Colecciones en Qdrant: {[coll.name for coll in qdrant_collections]}")
    except Exception as e:
        logger.error(f"Error al conectar con Qdrant: {e}")
        raise RuntimeError("No se pudo conectar con Qdrant")

    # Cargar colecciones desde JSON locales
    for dataset_file in knowledge_path.glob('*.json'):
        logger.info(f"Found JSON file: {dataset_file}")
        try:
            with open(dataset_file, 'r') as f:
                dataset_data = json.load(f)
            collection = Collection.from_dict(dataset_data)
            if not vdb._connection.collection_exists(collection.title):
                vdb.create_collection(collection, progress_bar=True)
                save_collection_metadata(collection, knowledge_path)
                logger.info(f"[+] Creada colección '{collection.title}' en Qdrant")
            else:
                logger.info(f"[+] La colección '{collection.title}' ya existe en Qdrant")
                vdb._collections[collection.title] = collection
        except Exception as e:
            logger.error(f"[!] Error loading dataset {dataset_file}: {e}")

    # Sincronizar colecciones desde Qdrant
    qdrant_collections = vdb._connection.get_collections().collections
    for qdrant_coll in qdrant_collections:
        coll_name = qdrant_coll.name
        if coll_name in vdb.collections:
            continue

        collection = Collection(
            collection_id=len(vdb._collections) + 1,
            title=coll_name,
            documents=[],
            topics=[],
            size=0
        )

        try:
            doc_names = set()
            all_topics = set()
            scroll_offset = None
            while True:
                points, next_offset = vdb._connection.scroll(
                    collection_name=coll_name,
                    limit=100,
                    offset=scroll_offset,
                    with_payload=True,
                    with_vectors=False
                )
                for point in points:
                    if 'doc_id' in point.payload:
                        doc_names.add(point.payload['doc_id'])
                    if 'topics' in point.payload and point.payload['topics']:
                        all_topics.update(point.payload['topics'])
                if not next_offset:
                    break
                scroll_offset = next_offset

            for doc_name in doc_names:
                collection.documents.append(Document(
                    name=doc_name,
                    content="",
                    topics=[Topic(t) for t in all_topics],
                    source_type="text"
                ))

            count_result = vdb._connection.count(collection_name=coll_name)
            collection.size = count_result.count
            collection.topics = [Topic(t) for t in all_topics]

            vdb._collections[coll_name] = collection
            save_collection_metadata(collection, knowledge_path)
            logger.info(f"[+] Sincronizada colección '{coll_name}' desde Qdrant con {len(doc_names)} documentos")
        except Exception as e:
            logger.error(f"Error sincronizando colección '{coll_name}' desde Qdrant: {e}")

    # Verificar estado de sincronización
    check_sync_status(vdb)

def save_collection_metadata(collection: Collection, knowledge_path: Path):
    """Guarda los metadatos de una colección en un archivo JSON."""
    metadata_file = knowledge_path / f"{collection.title}.json"
    try:
        with open(metadata_file, 'w') as f:
            json.dump(collection.to_dict(), f, indent=2)
        logger.info(f"Metadatos guardados en {metadata_file}")
    except Exception as e:
        logger.error(f"Error guardando metadatos de '{collection.title}': {e}")

def check_sync_status(vdb: EnhancedStore):
    local_collections = list(vdb.collections.keys())
    qdrant_collections = [coll.name for coll in vdb._connection.get_collections().collections]
    logger.info(f"Colecciones locales: {local_collections}")
    logger.info(f"Colecciones en Qdrant: {qdrant_collections}")
    missing_in_local = set(qdrant_collections) - set(local_collections)
    missing_in_qdrant = set(local_collections) - set(qdrant_collections)
    if missing_in_local:
        logger.warning(f"Colecciones en Qdrant pero no locales: {missing_in_local}")
    if missing_in_qdrant:
        logger.warning(f"Colecciones locales pero no en Qdrant: {missing_in_qdrant}")

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