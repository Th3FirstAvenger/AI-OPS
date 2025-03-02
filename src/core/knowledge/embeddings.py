"""
Embeddings and reranking functionality using Ollama models
instead of relying on Hugging Face.
"""
import httpx
from typing import List, Dict, Any, Optional, Union
from src.config import RAG_SETTINGS
from src.utils import get_logger

logger = get_logger(__name__)

class OllamaEmbeddings:
    """Client for Ollama embeddings."""
    
    def __init__(
        self,
        model: str = RAG_SETTINGS.EMBEDDING_MODEL,
        inference_endpoint: str = RAG_SETTINGS.EMBEDDING_URL,
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        client_timeout: int = 30
    ):
        """
        Initialize the Ollama embeddings client.
        
        Args:
            model: The embedding model to use (must be available in Ollama)
            inference_endpoint: The Ollama API endpoint
            api_key: API key (not required for Ollama)
            dimensions: Optional embedding dimensions to request
            client_timeout: HTTP client timeout in seconds
        """
        self.model = model
        self.inference_endpoint = inference_endpoint
        self.api_key = api_key  # Not used with Ollama but kept for API compatibility
        self.dimensions = dimensions
        self.client_timeout = client_timeout
        self.client = httpx.Client(timeout=client_timeout)
        
        # Validate that the model is available
        self._validate_model()
    
    def _validate_model(self) -> bool:
        """Check if the embedding model is available in Ollama."""
        try:
            response = self.client.get(f"{self.inference_endpoint}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name") for model in models]
                
                if self.model not in model_names:
                    logger.warning(f"Model {self.model} not found in Ollama. Available models: {model_names}")
                    logger.warning("You may need to pull the model with: ollama pull nomic-embed-text")
                    return False
                return True
            else:
                logger.error(f"Failed to get models from Ollama: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error checking Ollama models: {str(e)}")
            return False
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text query.
        
        Args:
            text: The text to embed
            
        Returns:
            The embedding vector
        """
        if not text.strip():
            logger.warning("Empty text provided for embedding")
            return []
            
        try:
            # Ollama embeddings endpoint
            url = f"{self.inference_endpoint}/api/embeddings"
            
            payload = {
                "model": self.model,
                "prompt": text
            }
            
            # Add dimensions if specified
            if self.dimensions:
                payload["dimensions"] = self.dimensions
                
            response = self.client.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("embedding", [])
            else:
                logger.error(f"Embedding request failed: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return []


class OllamaReranker:
    """Client for Ollama-based reranking using LLM."""
    
    def __init__(
        self,
        model: str = RAG_SETTINGS.RERANKER_MODEL,
        inference_endpoint: str = RAG_SETTINGS.EMBEDDING_URL,
        api_key: Optional[str] = None,
        client_timeout: int = 60,
        prompt_path: str = None
    ):
        """
        Initialize the Ollama reranker.
        
        Args:
            model: The LLM model to use for reranking (must be available in Ollama)
            inference_endpoint: The Ollama API endpoint
            api_key: API key (not required for Ollama)
            client_timeout: HTTP client timeout in seconds
            prompt_path: Path to the reranker prompt file (optional)
        """
        self.model = model
        self.inference_endpoint = inference_endpoint
        self.api_key = api_key
        self.client_timeout = client_timeout
        self.client = httpx.Client(timeout=client_timeout)
        
        # Load reranker prompt
        if prompt_path is None:
            from pathlib import Path
            default_prompt_path = Path(__file__).parent.parent.parent / "agent" / "architectures" / "default" / "prompts" / "reranker"
            if default_prompt_path.exists():
                with open(str(default_prompt_path), 'r', encoding='utf-8') as f:
                    self.reranker_prompt_template = f.read()
            else:
                # Fallback to default prompt if file doesn't exist
                self.reranker_prompt_template = """
                Please evaluate the relevance of the following documents to the user query: '{query}'

                Documents:
                {documents}

                Based on relevance to the query, rank the documents from most relevant to least relevant.
                Return only a comma-separated list of document indices in order of relevance (most relevant first).
                Example format: 2,0,4,1,3
                Answer: 
                """
        else:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.reranker_prompt_template = f.read()
    
    def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to the query using an LLM.
        
        Args:
            query: The user query
            documents: List of document dictionaries with at least a 'content' field
            top_n: Number of top documents to return
            
        Returns:
            List of reranked documents with scores
        """
        if not documents:
            return []
            
        if len(documents) <= 1:
            # No need to rerank a single document
            return documents
            
        try:
            # Create a prompt for the LLM to evaluate relevance
            prompt = self._create_reranking_prompt(query, documents)
            
            # Call Ollama API
            url = f"{self.inference_endpoint}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for more deterministic results
                }
            }
            
            response = self.client.post(url, json=payload)
            
            if response.status_code != 200:
                logger.error(f"Reranking request failed: {response.status_code} - {response.text}")
                return documents[:top_n]  # Return original order capped at top_n
                
            result = response.json()
            response_text = result.get("response", "")
            
            # Parse the reranking results
            reranked_indices = self._parse_reranking_response(response_text)
            
            # Reorder documents based on the indices
            reranked_documents = []
            for idx in reranked_indices:
                if 0 <= idx < len(documents):
                    doc = documents[idx].copy()
                    # Add a rerank score (inverse of position)
                    doc["rerank_score"] = 1.0 - (float(reranked_indices.index(idx)) / len(reranked_indices))
                    reranked_documents.append(doc)
            
            # Fill with any remaining documents in original order
            original_indices = set(range(len(documents)))
            used_indices = set(reranked_indices)
            unused_indices = original_indices - used_indices
            
            for idx in unused_indices:
                if len(reranked_documents) >= top_n:
                    break
                doc = documents[idx].copy()
                doc["rerank_score"] = 0.0  # Lowest score
                reranked_documents.append(doc)
            
            return reranked_documents[:top_n]
            
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            # Fallback to original order
            return documents[:top_n]
    
    
    def _create_reranking_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Create a prompt for the LLM to evaluate document relevance."""
        # Prepare documents text
        docs_text = ""
        for i, doc in enumerate(documents):
            content = doc.get("content", "").strip()
            # Truncate long documents
            if len(content) > 500:
                content = content[:500] + "..."
            docs_text += f"[{i}] {content}\n\n"
        
        # Check if prompt has placeholders
        if "{query}" in self.reranker_prompt_template and "{documents}" in self.reranker_prompt_template:
            # Use template with placeholders
            return self.reranker_prompt_template.format(query=query, documents=docs_text)
        else:
            # Use template as prefix and append query and documents
            return (
                f"{self.reranker_prompt_template}\n\n"
                f"User query: \"{query}\"\n\n"
                f"Documents:\n{docs_text}\n"
                f"Answer: "
            )
    
    def _parse_reranking_response(self, response: str) -> List[int]:
        """Parse the LLM's response to get reranking indices."""
        try:
            # Clean up the response text
            response = response.strip()
            
            # Handle possible formats
            if "," in response:
                # Comma-separated format (preferred)
                indices = [int(idx.strip()) for idx in response.split(",") if idx.strip().isdigit()]
            else:
                # Space-separated or other format
                indices = [int(idx.strip()) for idx in response.split() if idx.strip().isdigit()]
            
            return indices
        except Exception as e:
            logger.error(f"Error parsing reranking response: {str(e)}, response: {response}")
            return []


def create_ollama_embedding_provider(
    model: str = "nomic-embed-text",
    inference_endpoint: str = "http://localhost:11434",
    dimensions: Optional[int] = None
) -> OllamaEmbeddings:
    """Create an Ollama embeddings provider with the specified configuration."""
    return OllamaEmbeddings(
        model=model,
        inference_endpoint=inference_endpoint,
        dimensions=dimensions
    )


def create_ollama_reranker(
    model: str = "qllama/bge-reranker-large",
    inference_endpoint: str = "http://localhost:11434"
) -> OllamaReranker:
    """Create an Ollama reranker with the specified configuration."""
    return OllamaReranker(
        model=model,
        inference_endpoint=inference_endpoint
    )