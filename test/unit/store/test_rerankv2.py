"""
Test script for the enhanced hybrid search functionality with explicit reranking
"""
import os
import json
import numpy as np
from pathlib import Path
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import necessary modules
from src.core.knowledge import EnhancedStore, Collection, Document, Topic
from src.core.knowledge.document_processor import DocumentProcessor
from src.core.knowledge.neural_reranker import NeuralReranker, OllamaReranker

# Configuration
QDRANT_URL = os.environ.get("RAG_URL", "http://localhost:6333")
OLLAMA_HOST = os.environ.get("ENDPOINT", "http://localhost:11434")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
RERANKER_MODEL = "qllama/bge-reranker-large"  # Model for reranking


def print_comparison(original_results, reranked_results, query):
    """Print a side-by-side comparison of original and reranked results"""
    print("\n" + "=" * 80)
    print(f"RESULTS COMPARISON FOR QUERY: '{query}'")
    print("=" * 80)
    print(f"{'ORIGINAL ORDER':<40} | {'RERANKED ORDER':<40}")
    print("-" * 80)
    
    max_items = max(len(original_results), len(reranked_results))
    for i in range(max_items):
        orig_text = f"{i+1}. {original_results[i][:30]}..." if i < len(original_results) else ""
        rerank_text = f"{i+1}. {reranked_results[i][:30]}..." if i < len(reranked_results) else ""
        print(f"{orig_text:<40} | {rerank_text:<40}")

def create_test_collections():
    """Create test collections with various topics and content"""
    print("Creating test collections...")
    
    # Create a base path for the store
    base_path = str(Path.home() / '.aiops')
    
    # Initialize the store
    store = EnhancedStore(
        base_path=base_path,
        embedding_url=OLLAMA_HOST,
        embedding_model=EMBEDDING_MODEL,
        url=QDRANT_URL,
        in_memory=True,  # Use in-memory for testing
        use_reranker=True,
        reranker_provider="ollama",
        reranker_model=RERANKER_MODEL
    )
    
    # Create test collections
    
    # Collection 1: Security Tools
    security_tools_docs = [
        Document(
            name="nmap_guide.txt",
            content="Nmap stealth scanning uses SYN packets (-sS) to avoid completed TCP connections. "
                    "Common commands include:\n- nmap -sS -T2 target\n- nmap -sS -Pn target\n- nmap -sS -A target",
            topics=[Topic("security"), Topic("network"), Topic("scanning")],
            source_type="text"
        ),
        Document(
            name="metasploit_intro.txt",
            content="Metasploit is a penetration testing framework that makes hacking simpler. "
                    "It contains exploits for many vulnerabilities and simplifies exploit development.",
            topics=[Topic("security"), Topic("exploitation"), Topic("penetration testing")],
            source_type="text"
        ),
        Document(
            name="wireshark_basics.txt",
            content="Wireshark is a network protocol analyzer that allows you to capture and inspect packets. "
                    "It is useful for debugging network issues and security analysis.",
            topics=[Topic("security"), Topic("network"), Topic("analysis")],
            source_type="text"
        )
    ]
    
    security_tools = Collection(
        collection_id=1,
        title="Security Tools",
        documents=security_tools_docs,
        topics=[Topic("security"), Topic("network"), Topic("scanning"), Topic("exploitation"), Topic("penetration testing"), Topic("analysis")]
    )
    
    # Collection 2: Web Security
    web_sec_docs = [
        Document(
            name="sql_injection.txt",
            content="SQL injection is an attack where malicious SQL code is inserted into database queries. "
                    "Common tests include adding ' OR 1=1 -- to input fields.",
            topics=[Topic("web"), Topic("database"), Topic("injection")],
            source_type="text"
        ),
        Document(
            name="xss_tutorial.txt",
            content="Cross-site scripting (XSS) allows attackers to inject client-side scripts into web pages. "
                    "To test for XSS, try injecting <script>alert('XSS')</script> into input fields.",
            topics=[Topic("web"), Topic("javascript"), Topic("injection")],
            source_type="text"
        ),
        Document(
            name="csrf_protection.txt",
            content="Cross-Site Request Forgery (CSRF) attacks trick users into executing unwanted actions. "
                    "Protection involves using anti-CSRF tokens in forms and checking referrer headers.",
            topics=[Topic("web"), Topic("javascript"), Topic("security")],
            source_type="text"
        )
    ]
    
    web_security = Collection(
        collection_id=2,
        title="Web Security",
        documents=web_sec_docs,
        topics=[Topic("web"), Topic("database"), Topic("injection"), Topic("javascript"), Topic("security")]
    )
    
    # Create collections in store
    try:
        store.create_collection(security_tools)
        print(f"Created 'Security Tools' collection with {len(security_tools_docs)} documents")
    except Exception as e:
        print(f"Error creating 'Security Tools' collection: {e}")
        
    try:
        store.create_collection(web_security)
        print(f"Created 'Web Security' collection with {len(web_sec_docs)} documents")
    except Exception as e:
        print(f"Error creating 'Web Security' collection: {e}")
    
    return store

def test_reranking_direct(store):
    """Directly test the reranking process to show before/after comparison"""
    print("\n" + "=" * 80)
    print("DIRECT RERANKING TEST")
    print("=" * 80)
    
    # Create a reranker
    reranker = OllamaReranker(model_name=RERANKER_MODEL, endpoint=OLLAMA_HOST)
    
    # Set up some test cases
    test_cases = [
        {
            "query": "network security analysis",
            "passages": [
                "Wireshark is a network protocol analyzer that allows you to capture and inspect packets.",
                "Nmap stealth scanning uses SYN packets (-sS) to avoid completed TCP connections.",
                "Metasploit is a penetration testing framework that makes hacking simpler.",
                "SQL injection is an attack where malicious SQL code is inserted into database queries."
            ]
        },
        {
            "query": "injection attacks prevention",
            "passages": [
                "SQL injection is an attack where malicious SQL code is inserted into database queries.",
                "Cross-site scripting (XSS) allows attackers to inject client-side scripts into web pages.",
                "Nmap stealth scanning uses SYN packets (-sS) to avoid completed TCP connections.",
                "Cross-Site Request Forgery (CSRF) attacks trick users into executing unwanted actions."
            ]
        }
    ]
    
    # Test each case
    for case in test_cases:
        query = case["query"]
        passages = case["passages"]
        
        print(f"\nQuery: '{query}'")
        print("Original passages order:")
        for i, passage in enumerate(passages):
            print(f"{i+1}. {passage[:60]}...")
        
        # Apply reranking
        print("\nReranking...")
        scores = reranker.rerank(query, passages)
        
        # Create reranked list
        reranked_passages = [(passage, score) for passage, score in zip(passages, scores)]
        reranked_passages.sort(key=lambda x: x[1], reverse=True)
        
        print("\nReranked passages:")
        for i, (passage, score) in enumerate(reranked_passages):
            print(f"{i+1}. [{score:.4f}] {passage[:60]}...")

def test_hybrid_search_with_reranking(store):
    """Test the hybrid search functionality with and without reranking"""
    print("\n" + "=" * 80)
    print("HYBRID SEARCH WITH RERANKING TESTS")
    print("=" * 80)
    
    # Test cases
    test_cases = [
        {
            "name": "Search by collection name - WITH vs WITHOUT reranking",
            "query": "network security tools",
            "params": {
                "collection_name": "Security Tools",
                "limit": 5
            }
        },
        {
            "name": "Search by topics - WITH vs WITHOUT reranking",
            "query": "injection vulnerabilities",
            "params": {
                "topics": ["injection", "web"],
                "limit": 5
            }
        }
    ]
    
    for case in test_cases:
        print(f"\n{case['name']}")
        query = case["query"]
        params = case["params"]
        
        # Run search WITHOUT reranking
        print(f"\nQuery: '{query}' (WITHOUT reranking)")
        results_no_rerank = store.hybrid_search(
            query=query,
            rerank=False,
            **params
        )
        
        # Run search WITH reranking
        print(f"\nQuery: '{query}' (WITH reranking)")
        results_with_rerank = store.hybrid_search(
            query=query,
            rerank=True,
            **params
        )
        
        # Compare results
        for coll_name in results_no_rerank.keys():
            if coll_name in results_with_rerank:
                print(f"\nResults for collection: {coll_name}")
                original = results_no_rerank[coll_name]
                reranked = results_with_rerank[coll_name]
                
                if original and reranked:
                    print_comparison(original, reranked, query)
                else:
                    print("No results to compare.")

def explain_reranking_process():
    """Explain the reranking process in detail"""
    print("\n" + "=" * 80)
    print("HOW RERANKING WORKS IN THE HYBRID SEARCH")
    print("=" * 80)
    
    explanation = """
1. Initial Retrieval (without reranking):
   - BM25 Search: Text-based search using keyword matching
   - Vector Search: Semantic search using embedding similarity
   - Results Combination: Merges results from both methods with weights

2. Reranking Process:
   - The initial combined results are sent to the reranker
   - The reranker calculates new relevance scores based on semantic similarity
   - Results are reordered based on these new scores
   - More contextually relevant results rise to the top

3. Benefits of Reranking:
   - Improves result quality by considering deeper semantic relationships
   - Handles nuances and context that initial retrieval might miss
   - Particularly valuable for ambiguous queries or conceptual searches

4. Implementation:
   - Done through the NeuralReranker/OllamaReranker classes
   - Uses the same embedding model by default (configurable)
   - Applied automatically when rerank=True (default) in hybrid_search
"""
    print(explanation)

if __name__ == "__main__":
    print("=" * 80)
    print("ENHANCED HYBRID SEARCH WITH RERANKING TEST")
    print("=" * 80)
    
    # Create test collections
    store = create_test_collections()
    
    # Give time for indexing
    print("\nWaiting for indexing to complete...")
    time.sleep(2)
    
    # Explain reranking process
    explain_reranking_process()
    
    # Test direct reranking
    test_reranking_direct(store)
    
    # Test hybrid search with reranking
    test_hybrid_search_with_reranking(store)
    
    print("\nTest complete.")