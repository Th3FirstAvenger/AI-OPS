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


    def _chunk_markdown(self, text: str) -> list[str]:
        """Divides text into complete sections based on Markdown headers."""
        # Updated regex: matches # to ###### with optional space
        header_pattern = r'(^#{1,6}\s?.*$)'
        headers = [(m.start(), m.end()) for m in re.finditer(header_pattern, text, re.MULTILINE)]
        
        # Debugging: Log detected headers
        print(f"Detected {len(headers)} headers")
        for i, (start, end) in enumerate(headers):
            print(f"Header {i+1}: {text[start:end]}")
        
        if not headers:
            print("No headers found, returning full text")
            return [text.strip()] if text.strip() else []
        
        chunks = []
        # Capture text before the first header
        if headers[0][0] > 0:
            initial_chunk = text[:headers[0][0]].strip()
            if initial_chunk:
                chunks.append(initial_chunk)
                print(f"Initial chunk: {initial_chunk[:100]}...")
        
        # Split into sections
        for i in range(len(headers)):
            start = headers[i][0]
            end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
            section = text[start:end].strip()
            if section:  # Only add non-empty sections
                chunks.append(section)
                print(f"Section {i+1}: {section[:100]}...")
        
        return chunks

    def _chunk_text_plain(self, text: str) -> list[str]:
        """Splits plain text as before (example with sentences)."""
        # Here would go your original logic, for example:
        chunks = []
        current_chunk = ""
        sentences = text.split('. ')  # Assuming splitting by sentences
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += (". " + sentence if current_chunk else sentence)
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return chunks
    
    def _chunk_text(self, text: str, source_type: str) -> list[str]:
        """Divides text into fragments based on source_type."""
        if source_type == "markdown":
            return self._chunk_markdown(text)
        else:
            return self._chunk_text_plain(text)
    
    def process_document(self, document: Document) -> List[ChunkInfo]:
        """Process a document into chunks with metadata"""
        # Select chunking strategy based on document type
        chunks = self._chunk_text(document.content, document.source_type)
        
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