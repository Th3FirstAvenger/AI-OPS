"""BM25 indexing system for keyword-based retrieval"""
from typing import List, Tuple, Dict, Any, Set
import re
import json
import logging
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

# Handle dependencies in a single consistent way
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    # Ensure NLTK data is available
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        logger.info("Downloading required NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    
    # Import rank_bm25 or use fallback
    try:
        from rank_bm25 import BM25Okapi
        logger.info("Using rank_bm25 package for BM25 indexing")
    except ImportError:
        logger.warning("rank_bm25 package not found. Using fallback implementation.")
        # Define a simple placeholder class
        class BM25Okapi:
            def __init__(self, corpus):
                self.corpus = corpus
            
            def get_scores(self, query):
                # Return uniform scores for development purposes
                return [0.5] * len(self.corpus)
                
except ImportError as e:
    logger.error(f"Error importing dependencies: {e}")
    raise

from src.core.knowledge.document_processor import ChunkInfo


class BM25Index:
    """BM25 index for a collection of document chunks"""
    
    def __init__(self):
        self.chunks: List[ChunkInfo] = []
        self.index = None
        self.tokenized_chunks = []
        self.stop_words = set(stopwords.words('english'))
    
    def add_chunks(self, chunks: List[ChunkInfo]):
        """Add chunks to the index"""
        self.chunks.extend(chunks)
        self._rebuild_index()
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for indexing with special handling for technical terms"""
        # Original text for reference
        original_text = text
        
        # Preserve technical patterns (commands with options, IP addresses, etc.)
        preserved_patterns = {}
        
        # Preserve commands with options (e.g., "nmap -sV", "sqlmap --dbs")
        cmd_patterns = re.finditer(r'\b\w+(?:\s+(?:-{1,2}\w+)+)+', text)
        for i, match in enumerate(cmd_patterns):
            pattern = match.group(0)
            placeholder = f"__CMD_{i}__"
            preserved_patterns[placeholder] = pattern
            text = text.replace(pattern, placeholder)
        
        # Preserve IP addresses
        ip_patterns = re.finditer(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', text)
        for i, match in enumerate(ip_patterns):
            pattern = match.group(0)
            placeholder = f"__IP_{i}__"
            preserved_patterns[placeholder] = pattern
            text = text.replace(pattern, placeholder)
            
        # Preserve paths and filenames
        path_patterns = re.finditer(r'(?:/[\w\-\.]+)+/?|(?:[\w\-]+\.[\w\-]+)', text)
        for i, match in enumerate(path_patterns):
            pattern = match.group(0)
            placeholder = f"__PATH_{i}__"
            preserved_patterns[placeholder] = pattern
            text = text.replace(pattern, placeholder)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters (but not in preserved patterns)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Restore preserved patterns
        for placeholder, original in preserved_patterns.items():
            tokens = [original.lower() if token == placeholder.lower() else token for token in tokens]
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return tokens
    
    def _rebuild_index(self):
        """Rebuild the BM25 index with current chunks"""
        if not self.chunks:
            logger.warning("Attempting to build index with no chunks")
            return
            
        # Tokenize all chunks
        logger.info(f"Tokenizing {len(self.chunks)} chunks for BM25 indexing")
        self.tokenized_chunks = [self._preprocess_text(chunk.text) for chunk in self.chunks]
        
        # Build the BM25 index
        self.index = BM25Okapi(self.tokenized_chunks)
        logger.info(f"BM25 index built successfully with {len(self.tokenized_chunks)} documents")
    
    def search(self, query: str, limit: int = 5, filter_topics: List[str] = None) -> List[Tuple[ChunkInfo, float]]:
        """Search for chunks matching the query"""
        if not self.index:
            logger.warning("Search attempted on uninitialized BM25 index")
            return []
        
        # Preprocess the query
        tokenized_query = self._preprocess_text(query)
        
        # Get scores from BM25
        scores = self.index.get_scores(tokenized_query)
        
        # Combine chunks with their scores
        results = [(self.chunks[i], scores[i]) for i in range(len(scores))]
        
        # Filter by topics if specified
        if filter_topics:
            filter_topics_lower = [t.lower() for t in filter_topics]
            results = [
                (chunk, score) for chunk, score in results 
                if any(topic.lower() in filter_topics_lower for topic in chunk.topics)
            ]
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results
        return results[:limit]

    def save_to_disk(self, path: str or Path):
        """Save the index state to disk in a recoverable format"""
        data = {
            'chunks': [chunk.to_dict() for chunk in self.chunks],
            'tokenized_chunks': self.tokenized_chunks
        }
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.info(f"BM25 index saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save BM25 index to {path}: {e}")
            return False
            
    @classmethod
    def load_from_disk(cls, path: str or Path):
        """Load an index from disk and rebuild it"""
        index = cls()
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Restore chunks
            index.chunks = [ChunkInfo.from_dict(chunk_data) for chunk_data in data['chunks']]
            
            # Restore tokenized chunks
            index.tokenized_chunks = data['tokenized_chunks']
            
            # Rebuild the index
            index.index = BM25Okapi(index.tokenized_chunks)
            
            logger.info(f"BM25 index loaded from {path} with {len(index.chunks)} chunks")
            return index
        except Exception as e:
            logger.error(f"Failed to load BM25 index from {path}: {e}")
            raise

class CollectionBM25Index:
    """Manages BM25 indexes for multiple collections"""
    
    def __init__(self):
        self.collection_indexes: Dict[str, BM25Index] = {}
    
    def add_collection(self, collection_name: str, chunks: List[ChunkInfo]):
        """Create or update index for a collection"""
        if collection_name not in self.collection_indexes:
            self.collection_indexes[collection_name] = BM25Index()
            logger.info(f"Created new BM25 index for collection '{collection_name}'")
        
        self.collection_indexes[collection_name].add_chunks(chunks)
        logger.info(f"Added {len(chunks)} chunks to BM25 index for collection '{collection_name}'")
    
    def search(self, collection_name: str, query: str, limit: int = 5, filter_topics: List[str] = None) -> List[Tuple[ChunkInfo, float]]:
        """Search a specific collection"""
        if collection_name not in self.collection_indexes:
            logger.warning(f"Collection '{collection_name}' not found in BM25 indexes")
            return []
        
        logger.info(f"Searching collection '{collection_name}' for: {query}")
        if filter_topics:
            logger.info(f"Filtering by topics: {filter_topics}")
            
        return self.collection_indexes[collection_name].search(query, limit, filter_topics)
    
    def get_collections(self) -> List[str]:
        """Get all collection names with BM25 indexes"""
        return list(self.collection_indexes.keys())
        
    def save_collection_index(self, collection_name: str, base_path: str or Path):
        """Save a specific collection index"""
        if collection_name not in self.collection_indexes:
            logger.warning(f"Cannot save: Collection '{collection_name}' not found in BM25 indexes")
            return False
            
        path = Path(base_path) / f"{collection_name}.json"
        return self.collection_indexes[collection_name].save_to_disk(path)
        
    def load_collection_index(self, collection_name: str, base_path: str or Path):
        path = Path(base_path) / f"{collection_name}.json"
        if not path.exists():
            logger.debug(f"Warning: BM25 index file for {collection_name} not found at {path}")
            return False
        try:
            self.collection_indexes[collection_name] = BM25Index.load_from_disk(path)
            return True
        except json.JSONDecodeError as e:
            logger.debug(f"Error: The JSON file for {collection_name} is invalid: {e}")
            return False
        except Exception as e:
            logger.debug(f"Unexpected error loading BM25 index for {collection_name}: {e}")
            return False