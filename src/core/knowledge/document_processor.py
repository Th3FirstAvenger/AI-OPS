"""Document processing utilities for RAG system"""
import re
import spacy
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# Try to load spaCy model, download if not available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

from src.core.knowledge.collections import Document, Topic

class ChunkInfo:
    """Container for chunk information including text and metadata"""
    def __init__(self, text: str, doc_id: str, chunk_id: int, topics: List[str] = None, metadata: Dict[str, Any] = None):
        self.text = text
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.topics = topics or []
        self.metadata = metadata or {}
    
    def to_dict(self):
        return {
            'text': self.text,
            'doc_id': self.doc_id,
            'chunk_id': self.chunk_id,
            'topics': self.topics,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            text=data['text'],
            doc_id=data['doc_id'],
            chunk_id=data['chunk_id'],
            topics=data.get('topics', []),
            metadata=data.get('metadata', {})
        )

class DocumentProcessor:
    """Processes documents for ingestion into the RAG system"""
    
    def __init__(self, max_chunk_size: int = 500, min_chunk_size: int = 200):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
    
    def process_document(self, document: Document) -> List[ChunkInfo]:
        """Process a document into chunks with metadata"""
        # Select chunking strategy based on document type
        chunks = self._chunk_text(document.content)
    
        # Create ChunkInfo objects with metadata
        chunk_infos = []
        for i, chunk_text in enumerate(chunks):
            chunk_infos.append(ChunkInfo(
                text=chunk_text,
                doc_id=document.name,
                chunk_id=i,
                topics=[topic.name for topic in document.topics],
                metadata=document.metadata
            ))
        
        return chunk_infos
        
    
    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text using spaCy's sentence boundaries with smart merging"""
        # Process the text with spaCy to get sentence boundaries
        doc = nlp(text)
        sentences = list(doc.sents)
        
        # Initialize chunks
        chunks = []
        current_chunk = ""
        
        # Process each sentence
        for sentence in sentences:
            sentence_text = sentence.text.strip()
            
            # Skip empty sentences
            if not sentence_text:
                continue
            
            # If adding this sentence would exceed chunk size and we already have content,
            # start a new chunk
            if len(current_chunk) + len(sentence_text) > self.max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.min_chunk_size)
                current_chunk = current_chunk[overlap_start:] + " " + sentence_text
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += " " + sentence_text
                else:
                    current_chunk = sentence_text
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


    @staticmethod
    def from_file(file_path: str, topics: List[str] = None) -> Document:
        """Create a Document from a file with automatic type detection"""
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists() or not path.is_file():
            raise ValueError(f"File not found: {file_path}")
        
        # Read file content
        content = path.read_text(encoding="utf-8")
        
        # Determine file type
        suffix = path.suffix.lower()
        source_type = "text"  # Default
        
        if suffix in [".md", ".markdown"]:
            source_type = "markdown"
            return Document.from_markdown(path.name, content, topics or [])
        elif suffix in [".txt"]:
            source_type = "text"
        
        # Create document with appropriate topics
        document = Document(
            name=path.name,
            content=content,
            topics=[Topic(t) for t in (topics or [])],
            source_type=source_type
        )
        
        return document