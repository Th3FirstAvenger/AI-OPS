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

def initialize_knowledge(vdb: EnhancedStore):
    """Used to initialize and keep updated the Knowledge Base.
    Already existing Collections will not be overwritten.
    :param vdb: the reference to the Knowledge Base"""
    available = EnhancedStore.get_available_datasets()
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
        tool_registry, # Accept the tool_registry as a parameter instead of importing it
        use_reranker: bool = True,
        reranker_provider: str = "ollama",
        reranker_model: str = "",
        reranker_confidence: float = 0.0
):
    """Initialize and load the RAG system"""
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

    # Now we use the tool_registry parameter that was passed in
    @tool_registry.register(
        description=f"""Search documents in the RAG Vector Database using advanced hybrid retrieval.
        Available collections are:
        {available_documents}
        You can filter by topics and specify the level of detail you need.
        """
    )
    def search_rag(rag_query: str, collection: str, topics: str = None, detail_level: str = "medium") -> str:
        """
        Search the knowledge base with advanced options.
        
        :param rag_query: The search query
        :param collection: The collection name to search in
        :param topics: Optional comma-separated list of topics to filter by (e.g., "Enumeration,SMB")
        :param detail_level: Amount of detail to return ("brief", "medium", or "detailed")
        :return: Retrieved information from the knowledge base
        """
        try:
            # Process parameters
            topic_list = [t.strip() for t in topics.split(',')] if topics else None
            
            # Determine limit based on detail level
            limit_map = {"brief": 1, "medium": 3, "detailed": 5}
            limit = limit_map.get(detail_level.lower(), 3)
            
            # Execute search with all enhancements
            results = store.hybrid_retrieve(
                query=rag_query,
                collection_name=collection,
                topics=topic_list,
                limit=limit,
                rerank=True
            )
            
            if not results:
                return "No relevant information found."
            
            # Format results based on detail level
            if detail_level.lower() == "brief":
                return results[0]
            
            # For medium and detailed, join with separators
            separator = "\n\n" + "-" * 40 + "\n\n"
            return separator.join(results)
            
        except Exception as e:
            error_msg = f"Error searching RAG system: {str(e)}"
            import logging
            logging.getLogger(__name__).error(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg
    
    return store