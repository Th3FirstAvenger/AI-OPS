import json
from pathlib import Path
from typing import Dict, Optional, List, Union, Any

import httpx
import ollama
import spacy
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
import networkx as nx
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.core.knowledge.embeddings import OllamaEmbeddings, OllamaReranker
from src.core.knowledge.collections import Collection, Document, Topic
from src.core.knowledge.collections import MarkdownParser, DocumentChunk, chunk
from src.config import RAG_SETTINGS
from src.utils import get_logger

logger = get_logger(__name__)
nlp = spacy.load("en_core_web_md")

# --- GraphRAG ---
class GraphRAG:
    """Graph-based RAG that builds and queries a knowledge graph from document chunks"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.embedding_model = OllamaEmbeddings(
            model=RAG_SETTINGS.EMBEDDING_MODEL, 
            inference_endpoint=RAG_SETTINGS.EMBEDDING_URL
        )
            
    def build_graph(self, chunks: List[DocumentChunk]):
        """Build a knowledge graph from document chunks"""
        for i, chunk in enumerate(chunks):
            node_id = f"chunk_{i}"
            chunk.node_id = node_id
            self.graph.add_node(node_id, text=chunk.text, metadata=chunk.metadata)
            if i > 0 and chunks[i-1].metadata["source"] == chunk.metadata["source"]:
                self.graph.add_edge(chunks[i-1].node_id, node_id, type="sequence")
            for j in range(i):
                if (chunks[j].metadata["source"] == chunk.metadata["source"] and
                    chunks[j].metadata.get("header_context") == chunk.metadata.get("header_context")):
                    self.graph.add_edge(chunks[j].node_id, node_id, type="same_section")
        self._add_semantic_edges(chunks)
    
    def _add_semantic_edges(self, chunks: List[DocumentChunk], threshold: float = 0.7):
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.embed_documents(texts)
        from sklearn.metrics.pairwise import cosine_similarity
        cosine_scores = cosine_similarity(np.array(embeddings))
        for i in range(len(chunks)):
            for j in range(i+1, len(chunks)):
                similarity = cosine_scores[i][j]
                if similarity > threshold:
                    self.graph.add_edge(chunks[i].node_id, chunks[j].node_id, type="semantic", weight=similarity)
                    self.graph.add_edge(chunks[j].node_id, chunks[i].node_id, type="semantic", weight=similarity)
    
    def query_graph(self, query: str, top_k: int = 3) -> List[Dict]:
        query_embedding = self.embedding_model.embed_query(query)
        from sklearn.metrics.pairwise import cosine_similarity
        node_scores = [(node, cosine_similarity([query_embedding], [[self.graph.nodes[node]["text"]]])[0][0])
                       for node in self.graph.nodes()]
        node_scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        visited = set()
        for node, score in node_scores[:top_k]:
            if node in visited:
                continue
            results.append({
                "text": self.graph.nodes[node]["text"],
                "metadata": self.graph.nodes[node]["metadata"],
                "relevance": score
            })
            visited.add(node)
            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited:
                    edge_data = self.graph.get_edge_data(node, neighbor)
                    results.append({
                        "text": self.graph.nodes[neighbor]["text"],
                        "metadata": self.graph.nodes[neighbor]["metadata"],
                        "relevance": score * edge_data.get("weight", 0.5),
                        "relation": edge_data.get("type", "related")
                    })
                    visited.add(neighbor)
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:top_k]

# --- HybridRetriever ---
class HybridRetriever:
    """Hybrid retrieval system combining BM25, FAISS, and neural reranking"""
    
    def __init__(self):
        self.chunks = []  # List of DocumentChunk
        self.bm25 = None
        self.faiss_index = None
        self.embedding_model = OllamaEmbeddings(
            model=RAG_SETTINGS.EMBEDDING_MODEL, 
            inference_endpoint=RAG_SETTINGS.EMBEDDING_URL
        )
        self.reranker_model = OllamaReranker(
            model=RAG_SETTINGS.RERANKER_MODEL, 
            inference_endpoint=RAG_SETTINGS.EMBEDDING_URL
        )
        self.nlp = spacy.load("en_core_web_md")
        self.graph_rag = GraphRAG()
        self.corpus_tokenized = []
        self.dimension = 768
    
    def add_documents(self, documents: List[Document], use_graph: bool = True):
        chunks = []
        for doc in documents:
            if doc.name.endswith('.md'):
                parser = MarkdownParser()
                md_chunks = parser.parse_markdown(doc.content, doc.name)
                chunks.extend(md_chunks)
            else:
                for i, text in enumerate(chunk(doc)):
                    chunks.append(DocumentChunk(
                        text=text,
                        metadata={
                            "source": doc.name,
                            "topic": str(doc.topic),
                            "chunk_id": i
                        }
                    ))
        self.chunks = chunks
        self._create_bm25_index()
        self._create_faiss_index()
        if use_graph:
            self.graph_rag.build_graph(chunks)
    
    def _create_bm25_index(self):
        if not self.chunks:
            logger.warning("No documents to index for BM25.")
            return
        self.corpus_tokenized = []
        for c in self.chunks:
            doc = self.nlp(c.text)
            tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
            self.corpus_tokenized.append(tokens)
            c.bm25_tokens = tokens
        self.bm25 = BM25Okapi(self.corpus_tokenized)
    
    def _create_faiss_index(self):
        embeddings = []
        for c in self.chunks:
            embedding = self.embedding_model.embed_query(c.text)
            c.embedding = embedding
            embeddings.append(embedding)
        embeddings_np = np.array(embeddings).astype('float32')
        self.faiss_index = faiss.IndexFlatL2(self.dimension)
        self.faiss_index.add(embeddings_np)
    
    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        if not self.chunks:
            logger.warning("No documents indexed for BM25 search.")
            return []
        if self.bm25 is None:
            logger.info("BM25 index not initialized; creating it now.")
            self._create_bm25_index()
            if self.bm25 is None:
                logger.error("Failed to create BM25 index.")
                return []
        doc = self.nlp(query)
        query_tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
        bm25_scores = self.bm25.get_scores(query_tokens)
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
        if not self.chunks:
            logger.warning("No documents indexed for FAISS search.")
            return []
        if self.faiss_index is None:
            logger.info("FAISS index not initialized; attempting to create it.")
            self._create_faiss_index()
            if self.faiss_index is None:
                logger.error("Failed to create FAISS index.")
                return []
        query_embedding = self.embedding_model.embed_query(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "chunk": self.chunks[idx],
                "text": self.chunks[idx].text,
                "metadata": self.chunks[idx].metadata,
                "score": float(1.0 / (1.0 + distances[0][i])),
                "method": "faiss"
            })
        return results
    
    def _combine_results(self, bm25_results: List[Dict], faiss_results: List[Dict], graph_results: List[Dict]) -> List[Dict]:
        combined = {}
        for result in bm25_results:
            chunk_id = result["metadata"]["source"] + "_" + str(result["metadata"]["chunk_id"])
            if chunk_id not in combined:
                combined[chunk_id] = result
                result["score_bm25"] = result["score"]
            else:
                combined[chunk_id]["score_bm25"] = result["score"]
        for result in faiss_results:
            chunk_id = result["metadata"]["source"] + "_" + str(result["metadata"]["chunk_id"])
            if chunk_id not in combined:
                combined[chunk_id] = result
                result["score_faiss"] = result["score"]
            else:
                combined[chunk_id]["score_faiss"] = result["score"]
                if result["score"] > combined[chunk_id]["score"]:
                    combined[chunk_id]["score"] = result["score"]
        for result in graph_results:
            chunk_id = result["metadata"]["source"] + "_" + str(result["metadata"].get("chunk_id", 0))
            if chunk_id not in combined:
                combined[chunk_id] = {
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "score": result["relevance"],
                    "score_graph": result["relevance"],
                    "method": "graph"
                }
            else:
                combined[chunk_id]["score_graph"] = result["relevance"]
                if result["relevance"] > combined[chunk_id]["score"]:
                    combined[chunk_id]["score"] = result["relevance"]
        return list(combined.values())
    
    def _rerank_results(self, query: str, results: List[Dict], top_k: int) -> List[Dict]:
        if not results:
            return []
        reranked_docs = self.reranker_model.rerank(
            query=query,
            documents=[{"content": r["text"], "metadata": r["metadata"]} for r in results],
            top_n=top_k
        )
        for result in results:
            rerank_score = next((doc.get("rerank_score", 0.0) for doc in reranked_docs 
                                  if doc["content"] == result["text"]), 0.0)
            result["score_rerank"] = rerank_score
            result["score"] = rerank_score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def retrieve(self, query: str, top_k: int = 10, use_graph: bool = True) -> List[Dict]:
        bm25_results = self._bm25_search(query, top_k)
        faiss_results = self._faiss_search(query, top_k)
        graph_results = self.graph_rag.query_graph(query, top_k=top_k // 2) if use_graph else []
        combined_results = self._combine_results(bm25_results, faiss_results, graph_results)
        reranked_results = self._rerank_results(query, combined_results, top_k=RAG_SETTINGS.DEFAULT_RERANK_TOP_K)
        return reranked_results

# --- QdrantStore ---
class QdrantStore:
    """Qdrant-based knowledge store with hybrid retrieval and markdown support"""
    
    def __init__(
        self,
        qdrant_url: str = RAG_SETTINGS.RAG_URL,
        qdrant_api_key: str = RAG_SETTINGS.RAG_API_KEY,
        embedding_model: str = RAG_SETTINGS.EMBEDDING_MODEL,
        use_hybrid: bool = True
    ):
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collections = {}
        self.hybrid_retriever = HybridRetriever() if use_hybrid else None
        self.collections_indexed = set()
        self.embedding_model_name = embedding_model
        self.embedding_model = OllamaEmbeddings(model=embedding_model, inference_endpoint=RAG_SETTINGS.EMBEDDING_URL)
        self.embedding_size = 768
    
    def create_collection(self, collection: Collection, progress_bar: bool = False):
        logger.info(f"📢 Attempting to create collection: {collection.title}")
        try:
            collection_info = self.client.get_collection(collection.title)
            if collection_info:
                logger.info(f"✅ Collection {collection.title} already exists in Qdrant")
                return
        except Exception as e:
            logger.warning(f"⚠️ Collection {collection.title} does not exist. Proceeding. Error: {e}")
        try:
            self.client.create_collection(
                collection_name=collection.title,
                vectors_config=models.VectorParams(size=self.embedding_size, distance=models.Distance.COSINE)
            )
            logger.info(f"✅ Created collection {collection.title} in Qdrant")
        except Exception as e:
            logger.error(f"❌ Error creating collection {collection.title}: {e}")
            return
        self.collections[collection.title] = collection
        if collection.documents:
            logger.info(f"📢 Uploading {len(collection.documents)} documents to {collection.title}")
            self._upload_documents_to_qdrant(collection.documents, collection.title)
            if self.hybrid_retriever:
                self.hybrid_retriever.add_documents(collection.documents, use_graph=True)
                self.collections_indexed.add(collection.title)
                logger.info(f"✅ Indexed collection {collection.title} in hybrid retriever")
    
    def _upload_documents_to_qdrant(self, documents: List[Document], collection_name: str):
        points = []
        for doc_idx, doc in enumerate(documents):
            if doc.name.endswith('.md'):
                parser = MarkdownParser()
                chunks = parser.parse_markdown(doc.content, doc.name)
            else:
                text_chunks = chunk(doc)
                chunks = [DocumentChunk(
                    text=text,
                    metadata={"source": doc.name, "topic": str(doc.topic), "chunk_id": i}
                ) for i, text in enumerate(text_chunks)]
            for chunk_idx, chunk in enumerate(chunks):
                embedding = np.array(self.embedding_model.embed_query(chunk.text))
                point_id = f"{doc_idx}_{chunk_idx}"
                point = models.PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={"text": chunk.text, "metadata": chunk.metadata}
                )
                points.append(point)
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(collection_name=collection_name, points=batch)
            logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1} to {collection_name}")
    
    def upload(self, document: Document, collection_name: str):
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")
        self._upload_documents_to_qdrant([document], collection_name)
        self.collections[collection_name].documents.append(document)
        if self.hybrid_retriever:
            self.hybrid_retriever.add_documents([document])
    
    def retrieve_from(
        self,
        query: str,
        collection_name: str,
        limit: int = 5,
        threshold: float = 0.5,
        use_hybrid: bool = True,
        use_graph: bool = True
    ) -> Union[list[str], None]:
        if not query:
            raise ValueError('Query cannot be empty')
        if collection_name not in self.collections:
            raise ValueError(f'Collection {collection_name} does not exist')
        if use_hybrid and self.hybrid_retriever:
            if collection_name not in self.collections_indexed and self.collections[collection_name].documents:
                self.hybrid_retriever.add_documents(self.collections[collection_name].documents)
                self.collections_indexed.add(collection_name)
            retrieval_results = self.hybrid_retriever.retrieve(query=query, top_k=limit, use_graph=use_graph)
            filtered_results = [result for result in retrieval_results if result["metadata"]["source"].startswith(collection_name)]
            if filtered_results:
                return [result["text"] for result in filtered_results]
        return self._qdrant_search(query, collection_name, limit, threshold)
    
    def _qdrant_search(self, query: str, collection_name: str, limit: int, threshold: float) -> Union[list[str], None]:
        query_embedding = self.embedding_model.embed_query(query)
        search_results = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=threshold
        )
        if not search_results:
            return None
        results = [point.payload.get("text") for point in search_results if point.payload.get("text")]
        return results if results else None
    
    def get_collection(self, name: str) -> Optional[Collection]:
        return self.collections.get(name)
    
    @staticmethod
    def get_available_datasets() -> List[Collection]:
        i = 0
        collections = []
        datasets_path = Path(Path.home() / '.aiops' / 'datasets')
        if not datasets_path.exists():
            return []
        for p in datasets_path.iterdir():
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
