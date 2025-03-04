# test_rag_system.py
import pytest
from src.core.knowledge import (
    EnhancedStore, DocumentProcessor, Document, 
    Topic, NeuralReranker, BM25Index, ChunkInfo
)

@pytest.fixture
def test_store():
    """Create a test store for testing"""
    return EnhancedStore(
        base_path="./test_store",
        in_memory=True,
        use_reranker=True
    )

def test_document_processing():
    """Test document processing functionality"""
    processor = DocumentProcessor()
    doc = Document(
        name="test.md",
        content="# Test\n\nContent",
        topics=[Topic("Test")],
        source_type="markdown"
    )
    chunks = processor.process_document(doc)
    assert len(chunks) > 0
    assert chunks[0].text.startswith("# Test")

def test_bm25_indexing():
    """Test BM25 indexing functionality"""
    index = BM25Index()
    chunks = [
        ChunkInfo("test content", "doc1", 0, ["test"]),
        ChunkInfo("other content", "doc2", 0, ["other"])
    ]
    index.add_chunks(chunks)
    results = index.search("test", limit=1)
    assert len(results) == 1
    assert results[0][0].doc_id == "doc1"

def test_hybrid_retrieval(test_store):
    """Test hybrid retrieval functionality"""
    # Create collection
    from src.core.knowledge import Collection
    collection = Collection(
        collection_id=1,
        title="test_collection",
        documents=[],
        topics=[]
    )
    test_store.create_collection(collection)
    
    # Add document
    doc = Document(
        name="test.md",
        content="This is a test document about cybersecurity",
        topics=[Topic("Security")],
        source_type="text"
    )
    test_store.upload(doc, "test_collection")
    
    # Test retrieval
    results = test_store.hybrid_retrieve(
        query="cybersecurity",
        collection_name="test_collection",
        limit=1
    )
    assert len(results) == 1
    assert "cybersecurity" in results[0]