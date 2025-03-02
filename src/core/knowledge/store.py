"""RAG Vector Database interface with support for both local storage and Qdrant"""
import json
from pathlib import Path
from typing import Dict, Optional, List, Union, Any

import httpx
import ollama
import spacy
import qdrant_client.http.exceptions
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from src.core.knowledge.embeddings import OllamaEmbeddings, OllamaReranker
import numpy as np

import numpy as np
from rank_bm25 import BM25Okapi
import networkx as nx

from src.core.llm import ProviderError
from src.core.knowledge.collections import Collection, Document, Topic
from src.utils import get_logger
from src.core.knowledge.collections import MarkdownParser
from src.core.knowledge.collections import DocumentChunk, chunk
from src.config import RAG_SETTINGS



# Initialize logging
logger = get_logger(__name__)

# Load spaCy model for text processing
nlp = spacy.load("en_core_web_md")

# Define reusable model names
RERANKER_MODEL = RAG_SETTINGS.RERANKER_MODEL
EMBEDDING_MODEL = RAG_SETTINGS.EMBEDDING_MODEL
RAG_URL = RAG_SETTINGS.RAG_URL
RAG_API_KEY = RAG_SETTINGS.RAG_API_KEY
USE_HYBRID = RAG_SETTINGS.USE_HYBRID
EMBEDDING_URL = RAG_SETTINGS.EMBEDDING_URL
IN_MEMORY = RAG_SETTINGS.IN_MEMORY


class GraphRAG:
    """Graph-based RAG that builds and queries a knowledge graph from document chunks"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL, inference_endpoint=EMBEDDING_URL)
            
    def build_graph(self, chunks: List[DocumentChunk]):
        """Build a knowledge graph from document chunks"""
        # Add nodes for each chunk
        for i, chunk in enumerate(chunks):
            node_id = f"chunk_{i}"
            chunk.node_id = node_id
            
            # Add the chunk as a node
            self.graph.add_node(
                node_id,
                text=chunk.text,
                metadata=chunk.metadata
            )
            
            # Connect to previous chunk from same document (sequence edge)
            if i > 0 and chunks[i-1].metadata["source"] == chunk.metadata["source"]:
                prev_node_id = chunks[i-1].node_id
                self.graph.add_edge(prev_node_id, node_id, type="sequence")
                
            # Connect chunks with same header context (semantic edge)
            for j in range(i):
                if (chunks[j].metadata["source"] == chunk.metadata["source"] and
                    chunks[j].metadata["header_context"] == chunk.metadata["header_context"]):
                    self.graph.add_edge(chunks[j].node_id, node_id, type="same_section")
        
        # Add semantic similarity edges between chunks
        self._add_semantic_edges(chunks)
        
    def _add_semantic_edges(self, chunks: List[DocumentChunk], threshold: float = 0.7):
        """Add edges based on semantic similarity between chunks"""
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.embed_documents(texts)

        from sklearn.metrics.pairwise import cosine_similarity
        embeddings_np = np.array(embeddings)
        cosine_scores = cosine_similarity(embeddings_np)
        
        for i in range(len(chunks)):
            for j in range(i+1, len(chunks)):
                similarity = cosine_scores[i][j]
                if similarity > threshold:
                    self.graph.add_edge(
                        chunks[i].node_id, 
                        chunks[j].node_id, 
                        type="semantic",
                        weight=similarity
                    )
                    self.graph.add_edge(
                        chunks[j].node_id, 
                        chunks[i].node_id, 
                        type="semantic",
                        weight=similarity
                    )
    
    def query_graph(self, query: str, top_k: int = 3) -> List[Dict]:
        """Query the graph for relevant information"""
        query_embedding = self.embedding_model.embed_query(query)
        
        node_scores = []
        for node in self.graph.nodes():
            node_text = self.graph.nodes[node]["text"]
            node_embedding = self.embedding_model.embed_query(node_text)
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([query_embedding], [node_embedding])[0][0]
            node_scores.append((node, similarity))
        
        node_scores.sort(key=lambda x: x[1], reverse=True)
                
        # Get top k nodes and their neighborhoods
        results = []
        visited_nodes = set()
        
        for node, score in node_scores[:top_k]:
            if node in visited_nodes:
                continue
                
            # Add the node itself
            results.append({
                "text": self.graph.nodes[node]["text"],
                "metadata": self.graph.nodes[node]["metadata"],
                "relevance": score
            })
            visited_nodes.add(node)
            
            # Add context from the neighborhood (1-hop neighbors)
            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited_nodes:
                    edge_data = self.graph.get_edge_data(node, neighbor)
                    edge_weight = edge_data.get("weight", 0.5)
                    
                    results.append({
                        "text": self.graph.nodes[neighbor]["text"],
                        "metadata": self.graph.nodes[neighbor]["metadata"],
                        "relevance": score * edge_weight,  # Scale by edge weight
                        "relation": edge_data.get("type", "related")
                    })
                    visited_nodes.add(neighbor)
        
        # Re-sort by relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)
        
        return results[:top_k]


class HybridRetriever:
    """Hybrid retrieval system combining BM25, FAISS, and neural reranking"""
    
    def __init__(self):
        self.chunks = []
        self.bm25 = None
        self.faiss_index = None
        self.embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL, inference_endpoint=EMBEDDING_URL)
        self.reranker_model = OllamaReranker(model=RERANKER_MODEL, inference_endpoint=EMBEDDING_URL)
        self.nlp = spacy.load("en_core_web_md")
        self.graph_rag = GraphRAG()
        self.corpus_tokenized = []
        self.dimension = 768  # Dimension of the embedding model
        
    def add_documents(self, documents: List[Document], use_graph: bool = True):
        """Process and index documents"""
        chunks = []
        for doc in documents:
            # Check if it's a markdown document
            if doc.name.endswith('.md'):
                markdown_parser = MarkdownParser()
                md_chunks = markdown_parser.parse_markdown(doc.content, doc.name)
                chunks.extend(md_chunks)
            else:
                # Use existing chunking for non-markdown documents
                text_chunks = chunk(doc)
                for i, text in enumerate(text_chunks):
                    chunks.append(DocumentChunk(
                        text=text,
                        metadata={
                            "source": doc.name,
                            "topic": str(doc.topic),
                            "chunk_id": i
                        }
                    ))
        
        # Store all chunks
        self.chunks = chunks
        
        # Create BM25 index
        self._create_bm25_index()
        
        # Create FAISS index
        self._create_faiss_index()
        
        # Build knowledge graph if requested
        if use_graph:
            self.graph_rag.build_graph(chunks)
    
    def _create_bm25_index(self):
        """Create BM25 index for lexical search"""
        # Tokenize all chunks
        self.corpus_tokenized = []
        for chunk in self.chunks:
            doc = self.nlp(chunk.text)
            tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
            self.corpus_tokenized.append(tokens)
            chunk.bm25_tokens = tokens
        
        # Create BM25 index
        self.bm25 = BM25Okapi(self.corpus_tokenized)
    
    def _create_faiss_index(self):
        """Create FAISS index for semantic search"""
        # Create embeddings for all chunks
        embeddings = []
        for chunk in self.chunks:
            embedding = self.embedding_model.encode(chunk.text)
            chunk.embedding = embedding
            embeddings.append(embedding)
        
        # Convert to numpy array
        embeddings_np = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        import faiss
        self.faiss_index = faiss.IndexFlatL2(self.dimension)
        self.faiss_index.add(embeddings_np)
    
    def retrieve(self, query: str, top_k: int = 10, use_graph: bool = True) -> List[Dict]:
        """Retrieve relevant chunks using hybrid retrieval"""
        # 1. Get BM25 results
        bm25_results = self._bm25_search(query, top_k)
        
        # 2. Get FAISS results
        faiss_results = self._faiss_search(query, top_k)
        
        # 3. Get Graph results if requested
        graph_results = []
        if use_graph:
            graph_results = self.graph_rag.query_graph(query, top_k=top_k//2)
        
        # 4. Combine results
        combined_results = self._combine_results(bm25_results, faiss_results, graph_results)
        
        # 5. Rerank results
        reranked_results = self._rerank_results(query, combined_results, top_k=DEFAULT_RERANK_TOP_K)
        
        return reranked_results
    
    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform BM25 search"""
        # Tokenize query
        doc = self.nlp(query)
        query_tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
        
        # Get BM25 scores
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k chunks
        top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
        
        results = []
        for i in top_indices:
            results.append({
                "chunk": self.chunks[i],
                "text": self.chunks[i].text,
                "metadata": self.chunks[i].metadata,
                "score": float(bm25_scores[i]),
                "method": "bm25"
            })
        
        return results
    
    def _faiss_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform FAISS semantic search"""
        # Encode query
        query_embedding = self.embedding_model.encode(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search in FAISS index
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "chunk": self.chunks[idx],
                "text": self.chunks[idx].text,
                "metadata": self.chunks[idx].metadata,
                "score": float(1.0 / (1.0 + distances[0][i])),  # Convert distance to similarity score
                "method": "faiss"
            })
        
        return results
    
    def _combine_results(self, bm25_results: List[Dict], faiss_results: List[Dict], 
                         graph_results: List[Dict]) -> List[Dict]:
        """Combine results from different retrieval methods"""
        # Use a dictionary to track unique chunks
        combined = {}
        
        # Process BM25 results
        for result in bm25_results:
            chunk_id = result["metadata"]["source"] + "_" + str(result["metadata"]["chunk_id"])
            if chunk_id not in combined:
                combined[chunk_id] = result
                result["score_bm25"] = result["score"]
            else:
                combined[chunk_id]["score_bm25"] = result["score"]
        
        # Process FAISS results
        for result in faiss_results:
            chunk_id = result["metadata"]["source"] + "_" + str(result["metadata"]["chunk_id"])
            if chunk_id not in combined:
                combined[chunk_id] = result
                result["score_faiss"] = result["score"]
            else:
                combined[chunk_id]["score_faiss"] = result["score"]
                # Update the maximum score
                if result["score"] > combined[chunk_id]["score"]:
                    combined[chunk_id]["score"] = result["score"]
        
        # Process Graph results
        for result in graph_results:
            chunk_id = result["metadata"]["source"] + "_" + str(result["metadata"]["chunk_id"])
            if chunk_id not in combined:
                # Convert graph result format to match others
                combined[chunk_id] = {
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "score": result["relevance"],
                    "score_graph": result["relevance"],
                    "method": "graph"
                }
            else:
                combined[chunk_id]["score_graph"] = result["relevance"]
                # Update the maximum score
                if result["relevance"] > combined[chunk_id]["score"]:
                    combined[chunk_id]["score"] = result["relevance"]
        
        # Convert back to list
        return list(combined.values())
    
    def _rerank_results(self, query: str, results: List[Dict], top_k: int) -> List[Dict]:
        """Rerank results using a cross-encoder model"""
        if not results:
            return []
            
        # Extract text from results
        reranked_docs = self.reranker_model.rerank(
            query=query,
            documents=[{"content": result["text"], "metadata": result["metadata"]} for result in results],
            top_n=top_k
        )
        
        for i, result in enumerate(results):
            reranked_idx = next((idx for idx, doc in enumerate(reranked_docs) 
                                if doc["content"] == result["text"]), None)
            
            if reranked_idx is not None:
                result["score_rerank"] = reranked_docs[reranked_idx].get("rerank_score", 0.0)
            else:
                result["score_rerank"] = 0.0
                
            result["score"] = result["score_rerank"]
        
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]


# Original Store class (kept for backward compatibility)
class Store:
    """Act as interface for Qdrant database.
    Manages Collections and implements the Upload/Retrieve operations."""

    def __init__(
        self,
        base_path: str,
        embedding_url: str = 'http://localhost:11434',
        embedding_model: str = 'nomic-embed-text',
        url: str = 'http://localhost:6333',
        in_memory: bool = IN_MEMORY,
    ):
        """
        :param embedding_url:
            The url of the Ollama server.
        :param embedding_model:
            The embedding model that will be used to embed the documents.
            Ollama embedding models are:

            - nomic-embed-text (Default)
            - mxbai-embed-large
            - all-minilm
        :param url:
            Url must be provided to specify where is deployed Qdrant.
            Note: it won't be used if `in_memory` is set to True.
        :param in_memory:
            Specifies whether the Qdrant database is loaded in memory
            or it is deployed on a specific endpoint.
        """
        self.in_memory = in_memory

        if in_memory:
            self._connection = QdrantClient(':memory:')
            self._collections: Dict[str: Collection] = {}
        else:
            self._connection = QdrantClient(url)
            self._metadata: Path = Path(base_path)
            if not self._metadata.exists():
                self._metadata.mkdir(parents=True, exist_ok=True)

            try:
                available = self.get_available_collections()
            except qdrant_client.http.exceptions.ResponseHandlingException as err:
                raise RuntimeError("Can't get Qdrant collections") from err

            if available:
                coll = dict(available)
            else:
                coll = {}
            self._collections: Dict[str: Collection] = coll

        
        ###
        # En src/core/knowledge/store.py, dentro del método __init__
        try:
            self._encoder = ollama.Client(host=embedding_url).embeddings
            self._embedding_model: str = embedding_model

            # noinspection PyProtectedMember
            try:
                self._embedding_size: int = len(
                    self._encoder(
                        self._embedding_model,
                        prompt='init'
                    )['embedding']
                )
            except (httpx.ConnectError, ollama._types.ResponseError) as err:
                # Si no podemos conectarnos a Ollama, usar un valor predeterminado
                self._embedding_size = 768  # Dimensión predeterminada para nomic-embed-text
                from src.utils import get_logger
                logger = get_logger(__name__)
                logger.warning(f"No se pudo conectar a Ollama: {err}. Usando modo limitado.")
        except Exception as e:
            from src.utils import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error inicializando embeddings: {e}")
            # Configuración de respaldo
            self._embedding_size = 768
            # Crear una función mock para _encoder
            def mock_encoder(model, prompt):
                # Devolver un embedding simulado
                import numpy as np
                return {"embedding": np.zeros(self._embedding_size).tolist()}
            self._encoder = mock_encoder

    def create_collection(
        self,
        collection: Collection,
        progress_bar: bool = False
    ):
        """Creates a new Qdrant collection, uploads the collection documents
        using `upload` and creates a metadata file for collection."""
        if collection.title in self.collections:
            return None

        try:
            self._connection.create_collection(
                collection_name=collection.title,
                vectors_config=models.VectorParams(
                    size=self._embedding_size,
                    distance=models.Distance.COSINE
                )
            )
        except UnexpectedResponse as err:
            raise RuntimeError("Can't upload collection") from err

        # upload documents (if present)
        self._collections[collection.title] = collection
        for document in collection.documents:
            self.upload(document, collection.title)

        # should do logging
        # print(f'Collection {collection.title}: '
        #       f'initialized with {len(collection.documents)} documents')

        # update metadata in production (i.e persistent qdrant)
        if not self.in_memory:
            self.save_metadata(collection)

    def upload(
        self,
        document: Document,
        collection_name: str
    ):
        """Performs chunking and embedding of a document
        and uploads it to the specified collection"""
        if not isinstance(collection_name, str):
            raise TypeError(f'Expected str for collection_name, found {type(collection_name)}')
        if collection_name not in self._collections:
            raise ValueError('Collection does not exist')

        # create the Qdrant data points
        doc_chunks = chunk(document)
        emb_chunks = [{
            'title': document.name,
            'topic': str(document.topic),
            'text': ch,
            'embedding': self._encoder(self._embedding_model, ch)['embedding']
        } for ch in doc_chunks]
        current_len = self._collections[collection_name].size

        points = [
            models.PointStruct(
                id=current_len + i,
                vector=item['embedding'],
                payload={
                    'text': item['text'],
                    'title': item['title'],
                    'topic': item['topic']
                }
            )
            for i, item in enumerate(emb_chunks)
        ]

        # upload Points to Qdrant and update Collection metadata
        self._connection.upload_points(
            collection_name=collection_name,
            points=points
        )

        # self._collections[collection_name].documents.append(document)
        self._collections[collection_name].size = current_len + len(emb_chunks)

    def retrieve_from(
        self,
        query: str,
        collection_name: str,
        limit: int = 3,
        threshold: int = 0.5
    ) -> list[str] | None:
        """Performs retrieval of chunks from the vector database.
        :param query:
            A natural language query used to search in the vector database.
        :param collection_name:
            The name of the collection where the search happens; the collection
            must exist inside the vector database.
        :param limit:
            Number of maximum results returned by the search.
        :param threshold:
            Minimum similarity score that must be satisfied by the search
            results.
        :return: list of chunks or None
        """
        if not query:
            raise ValueError('Query cannot be empty')
        if collection_name not in self._collections.keys():
            raise ValueError(f'Collection {collection_name} does not exist')

        hits = self._connection.search(
            collection_name=collection_name,
            query_vector=self._encoder(self._embedding_model, query)['embedding'],
            limit=limit,
            score_threshold=threshold
        )
        results = [points.payload['text'] for points in hits]
        return results if results else None

    def save_metadata(self, collection: Collection):
        """Saves the collection metadata to the Store knowledge path.
        See [Collection.to_json_metadata](./collections.py)"""
        file_name = collection.title \
            if collection.title.endswith('.json') \
            else collection.title + '.json'
        new_file = str(Path(self._metadata / 'knowledge' / file_name))
        collection.to_json_metadata(new_file)

    def get_available_collections(self) -> Optional[dict[str, Collection]]:
        """Query qdrant for available collections in the database, then loads
        the metadata about the collections from USER/.aiops/knowledge."""
        if self.in_memory:
            return None

        # get collection names from qdrant
        available = self._connection.get_collections()
        names = []
        for collection in available.collections:
            names.append(collection.name)
        collections = []

        # get collection metadata from knowledge folder
        for p in Path(self._metadata / 'knowledge').iterdir():
            if not (p.exists() and p.suffix == '.json' and p.name in names):
                continue

            with open(str(p), 'r', encoding='utf-8') as fp:
                data = json.load(fp)

                docs = []
                for doc in data['documents']:
                    docs.append(Document(
                        name=doc['name'],
                        content=doc['content'],
                        topic=Topic(doc['topic'])
                    ))

                collections.append(Collection(
                    collection_id=data['id'],
                    title=data['title'],
                    documents=docs,
                    topics=[Topic(topic) for topic in data['topics']]
                ))

        return zip(names, collections)

    @staticmethod
    def get_available_datasets() -> list[Collection]:
        """Searches in USER/.aiops/datasets/ directory for available collections"""
        i = 0
        collections = []
        datasets_path = Path(Path.home() / '.aiops' / 'datasets')
        if not datasets_path.exists():
            return []

        for p in datasets_path.iterdir():
            # `i` is not incremented for each directory entry
            if not (p.is_file() and p.suffix == '.json'):
                if p.exists():
                    p.unlink()
                continue

            with open(p, 'r', encoding='utf-8') as fp:
                data = json.load(fp)

            topics = []
            documents = []
            for item in data:
                topic = Topic(item['category'])
                document = Document(
                    name=item['title'],
                    content=item['content'],
                    topic=topic
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


# New QdrantStore class implementation
class QdrantStore:
    """Qdrant-based knowledge store with hybrid retrieval and markdown support"""
    
    def __init__(
        self,
        qdrant_url: str = RAG_SETTINGS.RAG_URL,
        qdrant_api_key: str = RAG_SETTINGS.RAG_API_KEY,
        embedding_model: str = RAG_SETTINGS.EMBEDDING_MODEL,
        use_hybrid: bool = True
    ):
        """Initialize the Qdrant-based knowledge store"""
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        
        self.collections = {}
        self.hybrid_retriever = HybridRetriever() if use_hybrid else None
        self.collections_indexed = set()
        self.embedding_model_name = embedding_model
        self.embedding_model = OllamaEmbeddings(model=embedding_model, inference_endpoint=EMBEDDING_URL)
        self.embedding_size = 768

    def create_collection(self, collection: Collection, progress_bar: bool = False):
        """Create a new collection in Qdrant and index it with the hybrid retriever"""
        logger.info(f"📢 Attempting to create collection: {collection.title}")

        # Check if collection already exists
        try:
            collection_info = self.client.get_collection(collection.title)
            if collection_info:
                logger.info(f"✅ Collection {collection.title} already exists in Qdrant")
                return  # No need to recreate it
        except Exception as e:
            logger.warning(f"⚠️ Collection {collection.title} does not exist. Proceeding with creation. Error: {str(e)}")

        # Try to create the collection
        try:
            self.client.create_collection(
                collection_name=collection.title,
                vectors_config=models.VectorParams(
                    size=self.embedding_size,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"✅ Created collection {collection.title} in Qdrant")
        except Exception as e:
            logger.error(f"❌ Error creating collection {collection.title}: {e}")
            return

        # Store collection information
        self.collections[collection.title] = collection

        # Index documents in Qdrant
        if collection.documents:
            logger.info(f"📢 Uploading {len(collection.documents)} documents to {collection.title}")
            
            self._upload_documents_to_qdrant(collection.documents, collection.title)
            
            if self.hybrid_retriever:
                self.hybrid_retriever.add_documents(collection.documents)
                self.collections_indexed.add(collection.title)
                logger.info(f"✅ Indexed collection {collection.title} in hybrid retriever")

        """def create_collection(self, collection: Collection, progress_bar: bool = False):
        # Create a new collection in Qdrant and index it with the hybrid retriever
        logger.info(f"Attempting to create collection: {collection.title}")

        # Create collection in Qdrant if it doesn't exist
        try:
            collection_info = self.client.get_collection(collection.title)
            logger.info(f"Collection {collection.title} already exists in Qdrant")
        except Exception:
            # Collection doesn't exist, create it
            self.client.create_collection(
                collection_name=collection.title,
                vectors_config=models.VectorParams(
                    size=self.embedding_size,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created collection {collection.title} in Qdrant")
        except Exception as e:
            logger.error(f"Error creating collection {collection.title}: {e}")
            return
        
        # Store collection information
        self.collections[collection.title] = collection
    
        # Index documents with Qdrant
        if collection.documents:
            self._upload_documents_to_qdrant(collection.documents, collection.title)
            
            # Also index with hybrid retriever if enabled
            if self.hybrid_retriever:
                self.hybrid_retriever.add_documents(collection.documents)
                self.collections_indexed.add(collection.title)"""

    def _upload_documents_to_qdrant(self, documents: List[Document], collection_name: str):
        """Upload documents to Qdrant collection"""
        points = []
        for doc_idx, doc in enumerate(documents):
            # Process the document to create chunks
            if doc.name.endswith('.md'):
                markdown_parser = MarkdownParser()
                chunks = markdown_parser.parse_markdown(doc.content, doc.name)
            else:
                # Use existing chunking for non-markdown documents
                text_chunks = chunk(doc)
                chunks = []
                for i, text in enumerate(text_chunks):
                    chunks.append(DocumentChunk(
                        text=text,
                        metadata={
                            "source": doc.name,
                            "topic": str(doc.topic),
                            "chunk_id": i
                        }
                    ))
            
            # Create points for Qdrant
            for chunk_idx, chunk in enumerate(chunks):
                # Create embedding
                embedding = np.array(self.embedding_model.embed_query(chunk.text))

                
                # Create point
                point_id = f"{doc_idx}_{chunk_idx}"
                point = models.PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "text": chunk.text,
                        "metadata": chunk.metadata
                    }
                )
                points.append(point)
        
        # Upload points to Qdrant in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch
            )
            logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1} to {collection_name}")

    def upload(self, document: Document, collection_name: str):
        """Upload a document to a collection and index it"""
        # Check if collection exists
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")
        
        # Upload document to Qdrant
        self._upload_documents_to_qdrant([document], collection_name)
        
        # Add document to the collection
        self.collections[collection_name].documents.append(document)
        
        # Index with hybrid retriever if enabled
        if self.hybrid_retriever:
            self.hybrid_retriever.add_documents([document])

    def retrieve_from(self, query: str, collection_name: str, limit: int = 5, 
                    threshold: float = 0.5, use_hybrid: bool = True, use_graph: bool = True) -> list[str] | None:
        """Enhanced retrieval function using hybrid retrieval system and Qdrant"""
        if not query:
            raise ValueError('Query cannot be empty')
        if collection_name not in self.collections:
            raise ValueError(f'Collection {collection_name} does not exist')
            
        # Use hybrid retrieval if requested and available
        if use_hybrid and self.hybrid_retriever:
            # Check if collection is indexed with hybrid retriever
            if collection_name not in self.collections_indexed and self.collections[collection_name].documents:
                # If not, index it now
                self.hybrid_retriever.add_documents(self.collections[collection_name].documents)
                self.collections_indexed.add(collection_name)
                
            # Perform hybrid retrieval
            retrieval_results = self.hybrid_retriever.retrieve(
                query=query, 
                top_k=limit,
                use_graph=use_graph
            )
            
            # Filter results based on collection
            filtered_results = [
                result for result in retrieval_results
                if result["metadata"]["source"].startswith(collection_name)
            ]
            
            # Return the text of the top results
            if filtered_results:
                return [result["text"] for result in filtered_results]
        
        # Fall back to direct Qdrant search
        return self._qdrant_search(query, collection_name, limit, threshold)

    def _qdrant_search(self, query: str, collection_name: str, limit: int, threshold: float) -> list[str] | None:
        """Search using Qdrant's vector similarity search"""
        # Encode query
        query_embedding = self.embedding_model.embed_query(query)
        
        # Search in Qdrant
        search_results = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=threshold
        )
        
        # Extract text from results
        if not search_results:
            return None
            
        results = []
        for result in search_results:
            text = result.payload.get("text")
            if text:
                results.append(text)
        
        return results if results else None
        
    @staticmethod
    def get_available_datasets() -> list[Collection]:
        """Searches in USER/.aiops/datasets/ directory for available collections"""
        i = 0
        collections = []
        datasets_path = Path(Path.home() / '.aiops' / 'datasets')
        if not datasets_path.exists():
            return []

        for p in datasets_path.iterdir():
            # `i` is not incremented for each directory entry
            if not (p.is_file() and p.suffix == '.json'):
                if p.exists():
                    p.unlink()
                continue

            with open(p, 'r', encoding='utf-8') as fp:
                data = json.load(fp)

            topics = []
            documents = []
            for item in data:
                topic = Topic(item['category'])
                document = Document(
                    name=item['title'],
                    content=item['content'],
                    topic=topic
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

        return collections

    def get_collection(self, name):
        """Get a collection by name"""
        if name not in self.collections:
            return None
        return self.collections[name]