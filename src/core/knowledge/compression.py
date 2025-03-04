"""Text compression utilities for RAG documents"""
import zlib
import base64
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def compress_text(text: str, compression_level: int = 9) -> str:
    """
    Compress text using zlib and encode as base64 string.
    
    Args:
        text: Text to compress
        compression_level: Compression level (1-9, 9 being highest)
        
    Returns:
        Base64-encoded compressed text
    """
    if not text:
        return ""
        
    try:
        # Convert text to bytes
        text_bytes = text.encode('utf-8')
        
        # Compress with zlib
        compressed = zlib.compress(text_bytes, compression_level)
        
        # Encode as base64 string
        result = base64.b64encode(compressed).decode('ascii')
        
        # Log compression ratio
        original_size = len(text_bytes)
        compressed_size = len(result)
        ratio = compressed_size / original_size
        logger.debug(f"Compressed text: {original_size} bytes -> {compressed_size} bytes ({ratio:.2%})")
        
        return result
    except Exception as e:
        logger.error(f"Compression failed: {e}")
        return text

def decompress_text(compressed_text: str) -> str:
    """
    Decompress text that was compressed with compress_text.
    
    Args:
        compressed_text: Base64-encoded compressed text
        
    Returns:
        Original text
    """
    if not compressed_text:
        return ""
        
    try:
        # Decode base64
        compressed = base64.b64decode(compressed_text)
        
        # Decompress with zlib
        decompressed = zlib.decompress(compressed)
        
        # Decode bytes to string
        result = decompressed.decode('utf-8')
        
        return result
    except Exception as e:
        logger.error(f"Decompression failed: {e}")
        return compressed_text

def compress_document_metadata(metadata: Dict[str, Any], compress_content: bool = True) -> Dict[str, Any]:
    """
    Compress document metadata, optionally including content.
    
    Args:
        metadata: Document metadata dictionary
        compress_content: Whether to compress the 'content' field
        
    Returns:
        Compressed metadata
    """
    if not metadata:
        return {}
        
    compressed = metadata.copy()
    
    # Compress content if present and requested
    if compress_content and 'content' in compressed and compressed['content']:
        compressed['content'] = compress_text(compressed['content'])
        compressed['content_compressed'] = True
    
    return compressed

def decompress_document_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decompress document metadata.
    
    Args:
        metadata: Compressed document metadata
        
    Returns:
        Decompressed metadata
    """
    if not metadata:
        return {}
        
    decompressed = metadata.copy()
    
    # Decompress content if it was compressed
    if 'content_compressed' in decompressed and decompressed.get('content_compressed') and 'content' in decompressed:
        decompressed['content'] = decompress_text(decompressed['content'])
        decompressed['content_compressed'] = False
    
    return decompressed