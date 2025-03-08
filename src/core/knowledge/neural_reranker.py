"""Neural reranking integration for enhanced retrieval"""
import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import ollama

logger = logging.getLogger(__name__)

class BaseReranker(ABC):
    """Abstract base class for reranker implementations"""
    
    @abstractmethod
    def rerank(self, query: str, passages: List[str]) -> List[float]:
        """Rerank passages based on relevance to query, returning scores"""
        pass
        
    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if the reranker is properly initialized"""
        pass

class HuggingFaceReranker(BaseReranker):
    """Neural reranker implementation using Hugging Face models"""
    
    def __init__(self, model_name: str = "qllama/bge-reranker-large"):
        self.model_name = model_name
        self.model = None
        self._initialized = False
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the HuggingFace CrossEncoder model"""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            self._initialized = True
            logger.info(f"HuggingFace reranker initialized with model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize HuggingFace reranker: {e}")
            self._initialized = False
    
    def rerank(self, query: str, passages: List[str]) -> List[float]:
        """Rerank passages using the CrossEncoder model"""
        if not self._initialized or not self.model:
            logger.warning("HuggingFace reranker not initialized, returning empty scores")
            return [0.0] * len(passages)
            
        try:
            # Create query-passage pairs
            pairs = [[query, passage] for passage in passages]
            
            # Get scores from model
            scores = self.model.predict(pairs)
            
            return [float(score) for score in scores]
        except Exception as e:
            logger.error(f"Error during HuggingFace reranking: {e}")
            return [0.0] * len(passages)
            
    @property
    def is_initialized(self) -> bool:
        return self._initialized

class OllamaReranker(BaseReranker):
    """Neural reranker implementation using Ollama models"""
    
    def __init__(self, model_name: str = "nomic-embed-text", endpoint: str = "http://localhost:11434"):
        self.model_name = model_name
        self.endpoint = endpoint
        self.client = None
        self._initialized = False
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Ollama client"""
        try:
            self.client = ollama.Client(host=self.endpoint)
            
            # Test the connection
            self.client.embeddings(model=self.model_name, prompt="test")
            self._initialized = True
            logger.info(f"Ollama reranker initialized with model: {self.model_name} at {self.endpoint}")
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama reranker: {e}")
            self._initialized = False
    
    def rerank(self, query: str, passages: List[str]) -> List[float]:
        """Rerank passages using Ollama embedding similarity"""
        if not self._initialized or not self.client:
            logger.warning("Ollama reranker not initialized, returning empty scores")
            return [0.0] * len(passages)
            
        try:
            # Get query embedding
            query_embedding = self.client.embeddings(model=self.model_name, prompt=query)['embedding']
            
            # Get passage embeddings and calculate similarity
            scores = []
            for passage in passages:
                try:
                    passage_embedding = self.client.embeddings(model=self.model_name, prompt=passage)['embedding']
                    
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, passage_embedding)
                    scores.append(similarity)
                except Exception as e:
                    logger.error(f"Error embedding passage: {e}")
                    scores.append(0.0)
            
            return scores
        except Exception as e:
            logger.error(f"Error during Ollama reranking: {e}")
            return [0.0] * len(passages)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
            
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
            
        return dot_product / (norm_vec1 * norm_vec2)
            
    @property
    def is_initialized(self) -> bool:
        return self._initialized

class NeuralReranker:
    """Provider-agnostic neural reranker for search results"""
    
    def __init__(self, provider: str = "ollama", 
                model_name: Optional[str] = None,
                endpoint: str = "http://localhost:11434",
                confidence_threshold: float = 0.0):
        """
        Initialize the neural reranker.
        
        Args:
            provider: "huggingface" or "ollama"
            model_name: Model name to use (provider-specific)
            endpoint: API endpoint for provider (if applicable)
            confidence_threshold: Minimum score difference to apply reranking
        """
        self.provider = provider.lower()
        self.confidence_threshold = confidence_threshold
        
        # Default model names by provider
        self.default_models = {
            "huggingface": "nomic-embed-text",
            "ollama": "qllama/bge-reranker-large"
        }
        
        # Use default model if none provided
        if not model_name:
            model_name = self.default_models.get(self.provider, self.default_models["ollama"])
        
        self.model_name = model_name
        self.endpoint = endpoint
        self.reranker = self._initialize_reranker()
    
    def _initialize_reranker(self) -> BaseReranker:
        """Initialize the appropriate reranker based on provider"""
        if self.provider == "ollama":
            return OllamaReranker(self.model_name, self.endpoint)
        else:  # Default to huggingface
            return HuggingFaceReranker(self.model_name)
    
    @property
    def initialized(self) -> bool:
        """Check if the reranker is properly initialized"""
        return self.reranker and self.reranker.is_initialized
    
    def rerank(self, query: str, results: List[Dict[str, Any]], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank a list of results based on semantic relevance to the query.
        
        Args:
            query: The search query
            results: A list of result dictionaries
            limit: Maximum number of results to return
            
        Returns:
            Reranked list of result dictionaries
        """
        if not self.initialized:
            logger.warning(f"{self.provider.capitalize()} reranker not initialized, returning original results")
            return results[:limit] if limit else results
        
        if not results:
            return []
        
        try:
            # Extract passages for reranking
            passages = [result['text'] for result in results]
            
            # Time the reranking operation
            start_time = time.time()
            logger.info(f"Reranking {len(passages)} results with {self.provider} model")
            
            # Get reranking scores
            scores = self.reranker.rerank(query, passages)
            
            # Log reranking time
            elapsed_time = time.time() - start_time
            logger.info(f"Reranking completed in {elapsed_time:.2f} seconds")
            
            # Store original ranking for comparison
            for i, result in enumerate(results):
                result['original_rank'] = i
            
            # Add scores to results
            for i, score in enumerate(scores):
                results[i]['rerank_score'] = score
                
            # Re-sort based on reranker scores
            reranked_results = sorted(results, key=lambda x: x.get('rerank_score', 0.0), reverse=True)
            
            # Apply confidence threshold if enabled
            if self.confidence_threshold > 0:
                # Check if reranking significantly changed the order
                significant_change = False
                for i, result in enumerate(reranked_results[:limit if limit else len(reranked_results)]):
                    original_idx = result['original_rank']
                    if abs(i - original_idx) > 1:  # Position changed by more than 1
                        significant_change = True
                        break
                
                if not significant_change:
                    logger.info("Reranking did not significantly change results order, using original order")
                    # Restore original order
                    return sorted(results, key=lambda x: x['original_rank'])[:limit] if limit else len(results)
            
            # Apply limit if specified
            if limit and limit < len(reranked_results):
                logger.info(f"Returning top {limit} reranked results")
                return reranked_results[:limit]
            
            logger.info(f"Returning {len(reranked_results)} reranked results")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error during neural reranking: {e}")
            logger.error("Falling back to original result order")
            import traceback
            traceback.print_exc()
            return results[:limit] if limit else results