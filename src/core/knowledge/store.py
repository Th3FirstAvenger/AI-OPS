"""Enhanced RAG Vector Database with hybrid retrieval capabilities"""
import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any

import httpx
import ollama
import spacy
import qdrant_client.http.exceptions
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.core.llm import ProviderError
from src.core.knowledge.collections import Collection, Document, Topic
from src.core.knowledge.document_processor import DocumentProcessor, ChunkInfo
from src.core.knowledge.bm25_index import CollectionBM25Index
from src.core.knowledge.batch_utils import batch_process

# Set up logging
logger = logging.getLogger(__name__)

class EnhancedStore:
    """Interface for Qdrant database with hybrid retrieval capabilities.
    Manages Collections and implements the Upload/Retrieve operations."""

    def __init__(
        self,
        base_path: str,
        embedding_url: str = 'http://localhost:11434',
        embedding_model: str = 'nomic-embed-text',
        url: str = 'http://localhost:6333',
        in_memory: bool = False,
        use_reranker: bool = True,
        reranker_provider: str = "ollama",
        reranker_model: Optional[str] = None,
        reranker_confidence: float = 0.0
    ):
        """
        Initialize the enhanced store with hybrid retrieval capabilities.
        
        Args:
            base_path: Base directory for storing metadata and indexes
            embedding_url: URL of the embedding service (Ollama)
            embedding_model: Model to use for embeddings
            url: URL of the Qdrant server
            in_memory: Whether to use in-memory storage instead of persistent
            use_reranker: Whether to enable neural reranking
            reranker_provider: Reranker provider ("ollama" or "huggingface")
            reranker_model: Model name for the reranker (provider-specific)
            reranker_confidence: Minimum confidence threshold for reranking
        """
        self.in_memory = in_memory
        self.base_path = base_path

        # Initialize Qdrant client
        if in_memory:
            self._connection = QdrantClient(':memory:')
            self._collections: Dict[str, Collection] = {}
        else:
            self._connection = QdrantClient(url)
            self._metadata_path = Path(base_path)
            if not self._metadata_path.exists():
                self._metadata_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created metadata directory at {self._metadata_path}")

            # Create knowledge directory if it doesn't exist
            knowledge_dir = self._metadata_path / 'knowledge'
            if not knowledge_dir.exists():
                knowledge_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created knowledge directory at {knowledge_dir}")

            # Create BM25 directory if it doesn't exist
            bm25_dir = self._metadata_path / 'knowledge' / 'bm25'
            if not bm25_dir.exists():
                bm25_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created BM25 index directory at {bm25_dir}")

            try:
                available = self.get_available_collections()
                logger.info(f"Found {len(list(available)) if available else 0} available collections")
            except qdrant_client.http.exceptions.ResponseHandlingException as err:
                logger.error(f"Failed to get Qdrant collections: {err}")
                raise RuntimeError("Can't get Qdrant collections") from err

            if available:
                coll = dict(available)
            else:
                coll = {}
            self._collections: Dict[str, Collection] = coll

        # Initialize Ollama client for embeddings
        try:
            self._encoder = ollama.Client(host=embedding_url).embeddings
            logger.info(f"Connected to embedding service at {embedding_url}")
        except Exception as e:
            logger.error(f"Failed to connect to embedding service: {e}")
            raise ProviderError(f"Failed to connect to embedding service: {e}") from e

        self._embedding_model: str = embedding_model
        logger.info(f"Using embedding model: {embedding_model}")

        # Get embedding dimension
        try:
            self._embedding_size: int = len(
                self._encoder(
                    self._embedding_model,
                    prompt='init'
                )['embedding']
            )
            logger.info(f"Embedding size: {self._embedding_size}")
        except (httpx.ConnectError, ollama._types.ResponseError) as err:
            logger.error(f"Failed to load embedding model: {err}")
            raise ProviderError("Can't load embedding model") from err
        
        # Initialize document processor
        self.document_processor = DocumentProcessor()
        
        # Initialize BM25 index manager
        self.bm25_index = CollectionBM25Index()
        
        # Initialize the neural reranker if requested
        self.use_reranker = use_reranker
        if use_reranker:
            from src.core.knowledge.neural_reranker import NeuralReranker
            self.reranker = NeuralReranker(
                provider=reranker_provider,
                model_name=reranker_model,
                endpoint=embedding_url,  # Reuse the embedding endpoint for Ollama
                confidence_threshold=reranker_confidence
            )
            if self.reranker.initialized:
                logger.info(f"Neural reranker initialized with provider: {reranker_provider}")
            else:
                logger.warning(f"Neural reranker initialization failed, reranking will be disabled")
        else:
            self.reranker = None
        
        # Load existing BM25 indexes
        self._load_bm25_indexes()
        
        # Initialize result cache
        self.result_cache = {}
        self.max_cache_size = 100
        self.cache_ttl = 3600  # 1 hour in seconds
        
        # Lazy loading tracking
        self._loaded_collections = set()

    def create_collection(
        self,
        collection: Collection,
        progress_bar: bool = False
    ):
        """Creates a new Qdrant collection, uploads the collection documents
        using `upload` and creates a metadata file for collection."""
        if collection.title in self.collections:
            logger.warning(f"Collection '{collection.title}' already exists")
            return None

        try:
            logger.info(f"Creating collection '{collection.title}' in Qdrant")
            self._connection.create_collection(
                collection_name=collection.title,
                vectors_config=models.VectorParams(
                    size=self._embedding_size,
                    distance=models.Distance.COSINE
                )
            )
        except UnexpectedResponse as err:
            logger.error(f"Failed to create collection '{collection.title}': {err}")
            raise RuntimeError(f"Can't create collection '{collection.title}'") from err

        # Upload documents (if present)
        self._collections[collection.title] = collection
        document_count = len(collection.documents)
        logger.info(f"Adding {document_count} documents to collection '{collection.title}'")
        
        for i, document in enumerate(collection.documents):
            if progress_bar and i % 10 == 0:
                logger.info(f"Uploading document {i+1}/{document_count}...")
            self.upload(document, collection.title)

        # Update metadata in production (i.e persistent qdrant)
        if not self.in_memory:
            self.save_metadata(collection)
            
        logger.info(f"Collection '{collection.title}' created successfully with {document_count} documents")
        return collection.title

    def upload(
        self,
        document: Document,
        collection_name: str
    ):
        """Performs chunking and embedding of a document
        and uploads it to the specified collection"""
        try:
            if not isinstance(collection_name, str):
                raise TypeError(f'Expected str for collection_name, found {type(collection_name)}')
            if collection_name not in self._collections:
                raise ValueError(f"Collection '{collection_name}' does not exist")
            
            logger.info(f"Processing document '{document.name}' for collection '{collection_name}'")
            
            # Process document into chunks with metadata
            chunks = self.document_processor.process_document(document)
            if not chunks:
                logger.warning(f"No chunks extracted from document '{document.name}'")
                return
                
            logger.info(f"Document '{document.name}' processed into {len(chunks)} chunks")
            
            # Add chunks to BM25 index
            try:
                self.bm25_index.add_collection(collection_name, chunks)
                if not self.in_memory:
                    self._save_bm25_index(collection_name)
                logger.info(f"Added chunks to BM25 index for collection '{collection_name}'")
            except Exception as e:
                logger.error(f"BM25 indexing failed for '{document.name}': {e}")
                logger.warning("Falling back to vector search only")
                import traceback
                traceback.print_exc()
            
            # Create embeddings for chunks
            try:
                chunk_embeddings = []
                for i, chunk in enumerate(chunks):
                    try:
                        embedding = self._encoder(self._embedding_model, chunk.text)['embedding']
                        chunk_embeddings.append({
                            'text': chunk.text,
                            'doc_id': chunk.doc_id,
                            'chunk_id': chunk.chunk_id,
                            'topics': chunk.topics,
                            'metadata': chunk.metadata,
                            'embedding': embedding
                        })
                        if (i+1) % 10 == 0:
                            logger.debug(f"Processed {i+1}/{len(chunks)} embeddings")
                    except Exception as chunk_err:
                        logger.error(f"Failed to embed chunk {i} of '{document.name}': {chunk_err}")
                
                if not chunk_embeddings:
                    raise ValueError(f"Failed to create any embeddings for '{document.name}'")
                    
                logger.info(f"Created {len(chunk_embeddings)} embeddings for '{document.name}'")
                
            except Exception as emb_err:
                logger.error(f"Embedding creation failed for '{document.name}': {emb_err}")
                raise RuntimeError(f"Embedding creation failed: {emb_err}")
                
            # Get current collection size for ID assignment
            current_len = self._collections[collection_name].size

            # Prepare points for Qdrant
            points = [
                models.PointStruct(
                    id=current_len + i,
                    vector=item['embedding'],
                    payload={
                        'text': item['text'],
                        'doc_id': item['doc_id'],
                        'chunk_id': item['chunk_id'],
                        'topics': item['topics'],
                        'metadata': item['metadata']
                    }
                )
                for i, item in enumerate(chunk_embeddings)
            ]

            # Upload Points to Qdrant and update Collection metadata
            try:
                self._connection.upload_points(
                    collection_name=collection_name,
                    points=points
                )
                logger.info(f"Uploaded {len(points)} points to Qdrant collection '{collection_name}'")
            except Exception as upload_err:
                logger.error(f"Failed to upload points to Qdrant: {upload_err}")
                raise RuntimeError(f"Failed to upload to Qdrant: {upload_err}")

            # Add the document to the collection's document list if not already present
            doc_names = [doc.name for doc in self._collections[collection_name].documents]
            if document.name not in doc_names:
                self._collections[collection_name].documents.append(document)
            
            # Update collection size
            self._collections[collection_name].size = current_len + len(chunks)
            logger.info(f"Collection '{collection_name}' size updated to {self._collections[collection_name].size}")
            
            # Update metadata
            if not self.in_memory:
                self.save_metadata(self._collections[collection_name])
                
            return True
            
        except Exception as e:
            logger.error(f"Error uploading document '{document.name}' to '{collection_name}': {e}")
            import traceback
            traceback.print_exc()
            raise

    def hybrid_retrieve(
        self,
        query: str,
        collection_name: str,
        topics: List[str] = None,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        limit: int = 5,
        rerank: bool = True
    ) -> List[str]:
        """
        Perform hybrid retrieval combining BM25 and vector search with optional reranking.
        
        Args:
            query: The search query
            collection_name: Name of the collection to search
            topics: Optional list of topics to filter results
            bm25_weight: Weight for BM25 scores in the combined ranking
            vector_weight: Weight for vector scores in the combined ranking
            limit: Maximum number of results to return
            rerank: Whether to apply neural reranking
            
        Returns:
            List of text chunks matching the query
        """
        try:
            if not query:
                raise ValueError('Query cannot be empty')
            if collection_name not in self._collections.keys():
                raise ValueError(f"Collection '{collection_name}' does not exist")
            
            # Create a cache key
            cache_key = f"{query}|{collection_name}|{','.join(topics or [])}|{limit}|{rerank}"
            
            # Check cache
            current_time = time.time()
            if cache_key in self.result_cache:
                cache_entry = self.result_cache[cache_key]
                if current_time - cache_entry['timestamp'] < self.cache_ttl:
                    logger.info(f"Cache hit for query: {query}")
                    return cache_entry['results']
            
            # Ensure collection is loaded
            self._ensure_collection_loaded(collection_name)
            
            # For reranking, get more initial results
            should_rerank = rerank and self.use_reranker and self.reranker and self.reranker.initialized
            initial_limit = limit * 3 if should_rerank else limit
            
            # BM25 retrieval
            bm25_results = self.bm25_index.search(
                collection_name=collection_name,
                query=query,
                limit=initial_limit,
                filter_topics=topics
            )
            
            logger.info(f"BM25 search returned {len(bm25_results)} results")
            
            # Vector retrieval
            vector_results = self._vector_retrieve(
                query=query,
                collection_name=collection_name,
                limit=initial_limit,
                filter_topics=topics
            )
            
            logger.info(f"Vector search returned {len(vector_results)} results")
            
            # Combine results with weights
            combined_results = self._merge_results(
                bm25_results=bm25_results,
                vector_results=vector_results,
                bm25_weight=bm25_weight,
                vector_weight=vector_weight,
                limit=initial_limit
            )
            
            logger.info(f"Merged search returned {len(combined_results)} results")
            
            # Apply reranking if enabled and available
            final_results = combined_results
            if should_rerank and combined_results:
                logger.info("Applying neural reranking")
                final_results = self.reranker.rerank(query, combined_results, limit)
            else:
                # Just apply the limit if no reranking
                final_results = combined_results[:limit]
            
            # Extract text from final results
            results = [result['text'] for result in final_results]
            
            # Update cache
            self.result_cache[cache_key] = {
                'timestamp': current_time,
                'results': results
            }
            
            # Trim cache if it grows too large
            if len(self.result_cache) > self.max_cache_size:
                self._trim_cache()
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _vector_retrieve(
        self,
        query: str,
        collection_name: str,
        limit: int = 5,
        filter_topics: List[str] = None
    ) -> List[Tuple[Dict, float]]:
        """Perform vector-based retrieval"""
        # Prepare filter if topics are specified
        filter_obj = None
        if filter_topics:
            filter_obj = models.Filter(
                must=[
                    models.FieldCondition(
                        key="topics",
                        match=models.MatchAny(
                            any=[topic.lower() for topic in filter_topics]
                        )
                    )
                ]
            )
        
        # Get query embedding
        query_vector = self._encoder(self._embedding_model, query)['embedding']
        
        # Search in Qdrant
        hits = self._connection.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=0.0,  # We'll filter by score later
            query_filter=filter_obj
        )
        
        # Format results
        results = [
            (
                {
                    'text': point.payload['text'],
                    'doc_id': point.payload['doc_id'],
                    'chunk_id': point.payload.get('chunk_id', 0),
                    'topics': point.payload.get('topics', []),
                    'metadata': point.payload.get('metadata', {})
                },
                point.score
            )
            for point in hits
        ]
        
        return results

    def _merge_results(
        self,
        bm25_results: List[Tuple[ChunkInfo, float]],
        vector_results: List[Tuple[Dict, float]],
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        limit: int = 5
    ) -> List[Dict]:
        """Merge and rank results from BM25 and vector search"""
        # Create a unified result dictionary using doc_id and chunk_id as keys
        unified_results = {}
        
        # Process BM25 results
        for chunk_info, score in bm25_results:
            key = f"{chunk_info.doc_id}_{chunk_info.chunk_id}"
            if key not in unified_results:
                unified_results[key] = {
                    'text': chunk_info.text,
                    'doc_id': chunk_info.doc_id,
                    'chunk_id': chunk_info.chunk_id,
                    'topics': chunk_info.topics,
                    'metadata': chunk_info.metadata,
                    'bm25_score': score,
                    'vector_score': 0.0,
                    'combined_score': bm25_weight * score
                }
        
        # Process vector results
        for chunk_dict, score in vector_results:
            key = f"{chunk_dict['doc_id']}_{chunk_dict['chunk_id']}"
            if key in unified_results:
                # Update existing result
                unified_results[key]['vector_score'] = score
                unified_results[key]['combined_score'] += vector_weight * score
            else:
                # Add new result
                unified_results[key] = {
                    'text': chunk_dict['text'],
                    'doc_id': chunk_dict['doc_id'],
                    'chunk_id': chunk_dict['chunk_id'],
                    'topics': chunk_dict.get('topics', []),
                    'metadata': chunk_dict.get('metadata', {}),
                    'bm25_score': 0.0,
                    'vector_score': score,
                    'combined_score': vector_weight * score
                }
        
        # Convert to list and sort by combined score
        results_list = list(unified_results.values())
        results_list.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return results_list[:limit]

    def _load_bm25_indexes(self):
        """Load existing BM25 indexes from disk"""
        if self.in_memory:
            return
        
        bm25_dir = Path(self.base_path) / 'knowledge' / 'bm25'
        if not bm25_dir.exists():
            bm25_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created BM25 index directory at {bm25_dir}")
            return
        
        # Load each index file
        for index_file in bm25_dir.glob('*.json'):
            try:
                collection_name = index_file.stem
                logger.info(f"Loading BM25 index for collection '{collection_name}'")
                self.bm25_index.load_collection_index(collection_name, bm25_dir)
            except Exception as e:
                logger.error(f"Error loading BM25 index for '{collection_name}': {e}")
                import traceback
                traceback.print_exc()

    def _save_bm25_index(self, collection_name: str):
        """Save BM25 index to disk"""
        if self.in_memory:
            return
        
        bm25_dir = Path(self.base_path) / 'knowledge' / 'bm25'
        if not bm25_dir.exists():
            bm25_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving BM25 index for collection '{collection_name}'")
        self.bm25_index.save_collection_index(collection_name, bm25_dir)

    def _ensure_collection_loaded(self, collection_name: str):
        """Ensure BM25 index for a collection is loaded (lazy loading)"""
        if collection_name in self._loaded_collections:
            return
            
        if collection_name not in self.bm25_index.collection_indexes:
            # Try to load from disk
            bm25_dir = Path(self.base_path) / 'knowledge' / 'bm25'
            if self.bm25_index.load_collection_index(collection_name, bm25_dir):
                logger.info(f"Lazy-loaded BM25 index for collection '{collection_name}'")
                self._loaded_collections.add(collection_name)
            else:
                logger.warning(f"No BM25 index found for collection '{collection_name}'")
        else:
            self._loaded_collections.add(collection_name)

    def _trim_cache(self):
        """Remove oldest entries from cache when it exceeds max size"""
        # Sort cache keys by timestamp
        sorted_keys = sorted(
            self.result_cache.keys(),
            key=lambda k: self.result_cache[k]['timestamp']
        )
        
        # Remove oldest half
        keys_to_remove = sorted_keys[:len(sorted_keys) // 2]
        for key in keys_to_remove:
            del self.result_cache[key]
            
        logger.info(f"Trimmed cache, removed {len(keys_to_remove)} entries")

    def save_metadata(self, collection: Collection):
        """Saves the collection metadata to the Store knowledge path."""
        if self.in_memory:
            return
            
        file_name = collection.title \
            if collection.title.endswith('.json') \
            else collection.title + '.json'
        new_file = str(Path(self._metadata_path / 'knowledge' / file_name))
        
        # Make sure document information is included
        collection_data = collection.to_dict()
        
        # Log and save
        logger.info(f"Saving metadata for collection '{collection.title}' to {new_file}")
        logger.info(f"Collection has {len(collection.documents)} documents and {len(collection.topics)} topics")
        
        with open(new_file, 'w', encoding='utf-8') as f:
            json.dump(collection_data, f, indent=2)

    def get_available_collections(self) -> Optional[Dict[str, Collection]]:
        """Query qdrant for available collections in the database, then loads
        the metadata about the collections from USER/.aiops/knowledge."""
        if self.in_memory:
            return None

        # get collection names from qdrant
        available = self._connection.get_collections()
        names = [collection.name for collection in available.collections]
        collections = []

        # get collection metadata from knowledge folder
        for p in Path(self._metadata_path / 'knowledge').glob('*.json'):
            if not (p.exists() and p.suffix == '.json' and p.stem in names):
                continue

            try:
                with open(str(p), 'r', encoding='utf-8') as fp:
                    data = json.load(fp)

                    docs = []
                    for doc in data['documents']:
                        docs.append(Document(
                            name=doc['name'],
                            content=doc['content'],
                            topics=[Topic(topic) for topic in doc.get('topics', [])],
                            source_type=doc.get('source_type', 'text'),
                            metadata=doc.get('metadata', {})
                        ))

                    collections.append(Collection(
                        collection_id=data['id'],
                        title=data['title'],
                        documents=docs,
                        topics=[Topic(topic) for topic in data['topics']]
                    ))
            except Exception as e:
                logger.error(f"Error loading collection metadata from {p}: {e}")
        return dict(zip(names, collections))
        return zip(names, collections)

    @staticmethod
    def get_available_datasets() -> List[Collection]:
        """Searches in USER/.aiops/datasets/ directory for available collections"""
        i = 0
        collections = []
        datasets_path = Path(Path.home() / '.aiops' / 'datasets')
        if not datasets_path.exists():
            return []

        for p in datasets_path.iterdir():
            # `i` is not incremented for each directory entry
            if not (p.is_file() and p.suffix == '.json'):
                continue

            try:
                with open(p, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)

                topics = []
                documents = []
                for item in data:
                    topic = Topic(item['category'])
                    document = Document(
                        name=item['title'],
                        content=item['content'],
                        topics=[topic],
                        source_type=item.get('source_type', 'text'),
                        metadata={k: v for k, v in item.items() if k not in ['title', 'content', 'category', 'source_type']}
                    )
                    topics.append(topic)
                    documents.append(document)

                collection = Collection(
                    collection_id=i,
                    title=p.name,
                    documents=documents,
                    topics=topics
                )

                collections.append(collection)
                i += 1
            except Exception as e:
                logger.error(f"Error loading dataset from {p}: {e}")

        return collections

    @property
    def collections(self):
        """All stored collections
        :return dict str: Collection
        """
        return self._collections

    def get_collection(self, name):
        """Get a collection by name"""
        if name not in self.collections:
            return None
        return self._collections[name]
    
    def upload(
        self,
        document: Document,
        collection_name: str,
        batch_size: int = 10
    ):
        """
        Performs chunking and embedding of a document
        and uploads it to the specified collection
        
        Args:
            document: Document to upload
            collection_name: Name of collection to upload to
            batch_size: Size of batches for processing embeddings
        """
        try:
            if not isinstance(collection_name, str):
                raise TypeError(f'Expected str for collection_name, found {type(collection_name)}')
            if collection_name not in self._collections:
                raise ValueError(f"Collection '{collection_name}' does not exist")
            
            logger.info(f"Processing document '{document.name}' for collection '{collection_name}'")
            
            # Process document into chunks with metadata
            chunks = self.document_processor.process_document(document)
            if not chunks:
                logger.warning(f"No chunks extracted from document '{document.name}'")
                return
                
            chunk_count = len(chunks)
            logger.info(f"Document '{document.name}' processed into {chunk_count} chunks")
            
            # Add chunks to BM25 index
            try:
                self.bm25_index.add_collection(collection_name, chunks)
                if not self.in_memory:
                    self._save_bm25_index(collection_name)
                logger.info(f"Added chunks to BM25 index for collection '{collection_name}'")
            except Exception as e:
                logger.error(f"BM25 indexing failed for '{document.name}': {e}")
                logger.warning("Falling back to vector search only")
                import traceback
                traceback.print_exc()
            
            
            def process_batch(chunk_batch):
                batch_embeddings = []
                for chunk in chunk_batch:
                    try:
                        embedding = self._encoder(self._embedding_model, chunk.text)['embedding']
                        batch_embeddings.append({
                            'text': chunk.text,
                            'doc_id': chunk.doc_id,
                            'chunk_id': chunk.chunk_id,
                            'topics': chunk.topics,
                            'metadata': chunk.metadata,
                            'embedding': embedding
                        })
                    except Exception as chunk_err:
                        logger.error(f"Failed to embed chunk {chunk.chunk_id} of '{document.name}': {chunk_err}")
                return batch_embeddings
                
            chunk_embeddings = batch_process(chunks, process_batch, batch_size)
            
            if not chunk_embeddings:
                logger.error(f"Failed to create any embeddings for '{document.name}'")
                return False
                
            logger.info(f"Created {len(chunk_embeddings)} embeddings for '{document.name}'")
            
            # Get current collection size for ID assignment
            current_len = self._collections[collection_name].size

            # Prepare points for Qdrant
            points = [
                models.PointStruct(
                    id=current_len + i,
                    vector=item['embedding'],
                    payload={
                        'text': item['text'],
                        'doc_id': item['doc_id'],
                        'chunk_id': item['chunk_id'],
                        'topics': item['topics'],
                        'metadata': item['metadata']
                    }
                )
                for i, item in enumerate(chunk_embeddings)
            ]

            # Upload Points to Qdrant and update Collection metadata
            try:
                self._connection.upload_points(
                    collection_name=collection_name,
                    points=points
                )
                logger.info(f"Uploaded {len(points)} points to Qdrant collection '{collection_name}'")
            except Exception as upload_err:
                logger.error(f"Failed to upload points to Qdrant: {upload_err}")
                raise RuntimeError(f"Failed to upload to Qdrant: {upload_err}")

            # Add the document to the collection's document list if not already present
            doc_names = [doc.name for doc in self._collections[collection_name].documents]
            if document.name not in doc_names:
                self._collections[collection_name].documents.append(document)
            
            # Update collection size
            self._collections[collection_name].size = current_len + len(chunks)
            logger.info(f"Collection '{collection_name}' size updated to {self._collections[collection_name].size}")
            
            # Update metadata
            if not self.in_memory:
                self.save_metadata(self._collections[collection_name])
                
            return True
            
        except Exception as e:
            logger.error(f"Error uploading document '{document.name}' to '{collection_name}': {e}")
            import traceback
            traceback.print_exc()
            raise