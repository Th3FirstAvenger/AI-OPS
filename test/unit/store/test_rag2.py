# test_rag_direct.py
import os
from dotenv import load_dotenv
from src.core.knowledge import load_rag
from src.config import RAG_SETTINGS

# Load environment variables
load_dotenv()

def test_rag():
    print("Initializing RAG system...")
    store = load_rag(
        rag_endpoint=RAG_SETTINGS.RAG_URL,
        in_memory=RAG_SETTINGS.IN_MEMORY,
        embedding_model=RAG_SETTINGS.EMBEDDING_MODEL,
        embedding_url=RAG_SETTINGS.EMBEDDING_URL,
        tool_registry=None,
        use_reranker=RAG_SETTINGS.USE_RERANKER
    )
    
    print(f"RAG initialized with {len(store.collections)} collections")
    for name, collection in store.collections.items():
        print(f"Collection: {name} with {len(collection.documents)} documents")
    
    query = "kerberoasting attack techniques"
    print(f"\nSearching for: {query}")
    
    results = store.hybrid_search(
        query=query,
        topics=["active-directory", "kerberos"],
        limit=3,
        rerank=True
    )
    
    for coll_name, texts in results.items():
        print(f"\nResults from {coll_name}:")
        for i, text in enumerate(texts):
            print(f"{i+1}. {text}")

if __name__ == "__main__":
    test_rag()