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
        if document.source_type == "markdown":
            chunks = self._chunk_technical_document(document.content)
        else:
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



    def process_markdown_file(file_path: str, topics: List[str]) -> Document:
        """Convierte un archivo Markdown en un objeto Document."""
        def parse_frontmatter(content):
            if not content.startswith('---'):
                return {}, content
            
            try:
                end_index = content.find('---', 3)
                if end_index == -1:
                    return {}, content
                    
                frontmatter_text = content[3:end_index].strip()
                actual_content = content[end_index + 3:].strip()
                
                metadata = {}
                for line in frontmatter_text.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
                        
                return metadata, actual_content
            except Exception:
                return {}, content
                
        with open(file_path, 'r', encoding='utf-8') as f:
            metadata, content = parse_frontmatter(f.read())
            
        return Document(
            name=Path(file_path).name,
            content=content,
            topics=[Topic(t) for t in topics],
            metadata=metadata
        )
        
    
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
    
    def _chunk_technical_document(self, text: str) -> List[str]:
        """Chunk technical document preserving command blocks and logical sections"""
        # Patterns to detect structural elements
        code_block_pattern = r'(```[^\n]*\n[\s\S]*?```|\$\s+.*|#\s+.*)'
        heading_pattern = r'(^|\n)(#{1,6}\s+[^\n]+)'
        numbered_step_pattern = r'(^|\n)(\d+\.\s+[^\n]+)'
        
        # Find all structural elements with their positions
        elements = []
        
        for pattern, element_type in [
            (code_block_pattern, 'code'),
            (heading_pattern, 'heading'),
            (numbered_step_pattern, 'step')
        ]:
            for match in re.finditer(pattern, text, re.MULTILINE):
                elements.append({
                    'type': element_type,
                    'start': match.start(),
                    'end': match.end(),
                    'text': match.group(0)
                })
        
        # Sort elements by position
        elements.sort(key=lambda x: x['start'])
        
        # Build chunks respecting structure
        chunks = []
        current_chunk = ""
        last_pos = 0
        
        for elem in elements:
            # Add text between elements
            if elem['start'] > last_pos:
                current_chunk += text[last_pos:elem['start']]
            
            # Add the element itself
            current_chunk += elem['text']
            last_pos = elem['end']
            
            # Check if we should start a new chunk
            if len(current_chunk) >= self.max_chunk_size and elem['type'] in ['heading', 'step']:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = ""
        
        # Add the final piece
        if last_pos < len(text):
            current_chunk += text[last_pos:]
            
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Handle overly large chunks
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.max_chunk_size * 1.5:
                # For overly large chunks, use sentence-based splitting as fallback
                final_chunks.extend(self._chunk_text(chunk))
            else:
                final_chunks.append(chunk)
                
        return final_chunks

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