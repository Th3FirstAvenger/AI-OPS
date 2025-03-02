"""Knowledge Module: contains vector database and documents classes"""
from pathlib import Path
from tool_parse import ToolRegistry

# Initialize RAG (basic version for now)
from src.core.knowledge.store import Store
from src.config import RAG_SETTINGS
from .collections import Collection, Topic
from .store import Store
__all__ = ['Collection', 'Topic', 'Store']
from src.core.knowledge.embeddings import OllamaEmbeddings, OllamaReranker
from src.core.knowledge.embeddings import create_ollama_embedding_provider, create_ollama_reranker

try:
    rag_store = Store(
        base_path=str(Path(Path.home() / '.aiops')),
        url=RAG_SETTINGS.RAG_URL,
        embedding_url=RAG_SETTINGS.EMBEDDING_URL,
        embedding_model=RAG_SETTINGS.EMBEDDING_MODEL,
        in_memory=RAG_SETTINGS.IN_MEMORY,
    )
except ConnectionError:
    # Crear un objeto Store con modo offline o con embeddings simulados
    from src.utils import get_logger
    logger = get_logger(__name__)
    logger.warning("Failed to connect to RAG, using offline mode")
    rag_store = Store(
        base_path=str(Path(Path.home() / '.aiops')),
        in_memory=True, 
    )
    


def initialize_knowledge(vdb: Store):
    """Used to initialize and keep updated the Knowledge Base.
    Already existing Collections will not be overwritten.
    :param vdb: the reference to the Knowledge Base"""
    available = Store.get_available_datasets()
    print(f"[+] Available Datasets ({[c.title for c in available]})")

    existing: list[str] = list(vdb.collections.keys())
    print(f"[+] Available Collections ({existing})")

    for collection in available:
        if collection.title not in existing:
            vdb.create_collection(collection, progress_bar=True)


def load_rag(
        rag_endpoint: str,
        in_memory: bool,
        embedding_model: str,
        embedding_url: str,
        tool_registry: ToolRegistry,
):
    store = Store(
        str(Path(Path.home() / '.aiops')),
        url=rag_endpoint,
        embedding_url=embedding_url,
        embedding_model=embedding_model,
        in_memory=in_memory
    )

    initialize_knowledge(store)
    available_documents = ''
    for cname, coll in store.collections.items():
        doc_topics = ", ".join([topic.name for topic in coll.topics])
        available_documents += f"- '{cname}': {doc_topics}\n"

    @tool_registry.register(
        description=f"""Search documents in the RAG Vector Database.
        Available collections are:
        {available_documents}
        """
    )
    def search_rag(rag_query: str, collection: str) -> str:
        """
        :param rag_query: what should be searched
        :param collection: the collection name
        """
        return '\n\n'.join(store.retrieve_from(rag_query, collection))
