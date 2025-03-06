from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from ollama import Client as OllamaClient
import numpy as np
import json
import os
from src.core.knowledge.neural_reranker import OllamaReranker 

# Configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "my_collection"
OLLAMA_HOST = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"  # Model for embeddings
RERANKER_MODEL = "qllama/bge-reranker-large"  # Model for reranking
LLM_MODEL = "deepseek-r1:14b"         # Model for generating responses

# Initialize clients
qdrant_client = QdrantClient(QDRANT_URL)
ollama_client = OllamaClient(host=OLLAMA_HOST)

# Generate sample documents in JSON
def generate_sample_documents():
    documents = [
        {"id": 1, "content": "Artificial intelligence is transforming the world with advances in machine learning and natural language processing."},
        {"id": 2, "content": "Large language models, like those developed by xAI, can generate human-like text and answer complex questions."},
        {"id": 3, "content": "Climate change is a global challenge that requires immediate action to reduce carbon emissions."},
        {"id": 4, "content": "Python programming is widely used in data science and AI development due to its simplicity and flexibility."},
        {"id": 5, "content": "RAG systems combine information retrieval and text generation to improve AI model responses."}
    ]
    with open("data.json", "w") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    return documents

# Load documents from JSON
def load_documents(json_path: str):
    with open(json_path, "r") as f:
        return json.load(f)

# Upload documents to Qdrant
def upload_documents(documents):
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        sample_embedding = ollama_client.embeddings(model=EMBEDDING_MODEL, prompt="test")["embedding"]
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"size": len(sample_embedding), "distance": "Cosine"}
        )
    
    for doc in documents:
        embedding = ollama_client.embeddings(model=EMBEDDING_MODEL, prompt=doc["content"])["embedding"]
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[{"id": doc["id"], "vector": embedding, "payload": doc}]
        )

# Initialize BM25
def initialize_bm25(documents):
    tokenized_docs = [doc["content"].split() for doc in documents]
    return BM25Okapi(tokenized_docs)

# Generate embeddings for the query
def embed_text(text: str):
    return ollama_client.embeddings(model=EMBEDDING_MODEL, prompt=text)["embedding"]

# Main hybrid search and reranking function
def hybrid_rag_search(query: str, bm25, documents):
    # Step 1: BM25 Search
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_k_indices = np.argsort(bm25_scores)[::-1][:100]  # Top 100 documents
    bm25_candidates = [documents[i] for i in top_k_indices]

    # Step 2: Qdrant Search
    query_vector = embed_text(query)
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=20  # Top 20 documents
    )
    qdrant_docs = [hit.payload for hit in search_results]

    # Combine candidates (BM25 + Qdrant)
    combined_docs = list({doc["id"]: doc for doc in bm25_candidates + qdrant_docs}.values())

    # Step 3: Reranking with OllamaReranker
    reranker = OllamaReranker(model_name=RERANKER_MODEL, endpoint=OLLAMA_HOST)
    passages = [doc["content"] for doc in combined_docs]
    scores = reranker.rerank(query, passages)

    # Associate scores with documents
    reranked_results = [{"content": doc["content"], "score": score} for doc, score in zip(combined_docs, scores)]

    # Sort by score
    reranked_results.sort(key=lambda x: x["score"], reverse=True)
    print("Reranked Results:")
    for i, result in enumerate(reranked_results):
        print(f"{i+1}. {result['content']} (Score: {result['score']:.3f})")
    top_results = reranked_results[:5]  # Top 5 documents

    # Step 4: Generate response with LLM
    context = "\n".join([result["content"] for result in top_results])
    prompt = f"Based on the following context, answer the query: {query}\n\nContext:\n{context}"
    print(f"Prompt: {prompt}")
    print("__________________________________________________________")
    print("Generating response with LLM...\n")
    llm_response = ollama_client.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    return llm_response["message"]["content"]

# System execution
if __name__ == "__main__":
    # Load or generate documents
    json_path = "data.json"
    if not os.path.exists(json_path):
        documents = generate_sample_documents()
    else:
        documents = load_documents(json_path)

    # Upload documents to Qdrant
    upload_documents(documents)

    # Initialize BM25
    bm25 = initialize_bm25(documents)

    # Test with a query
    query = "What is artificial intelligence?"
    response = hybrid_rag_search(query, bm25, documents)
    print(f"Query: {query}")
    print(f"Response: {response}")

"""
(AI-OPS) (3.11.6) λ ~/**/ retrive_test_2* python3 test/unit/store/test_rerank.py
Reranked Results:
1. Artificial intelligence is transforming the world with advances in machine learning and natural language processing. (Score: 1.000)
2. Python programming is widely used in data science and AI development due to its simplicity and flexibility. (Score: 0.988)
3. Large language models, like those developed by xAI, can generate human-like text and answer complex questions. (Score: 0.937)
4. Climate change is a global challenge that requires immediate action to reduce carbon emissions. (Score: 0.921)
5. RAG systems combine information retrieval and text generation to improve AI model responses. (Score: 0.815)
Prompt: Based on the following context, answer the query: What is artificial intelligence?

Context:
Artificial intelligence is transforming the world with advances in machine learning and natural language processing.
Python programming is widely used in data science and AI development due to its simplicity and flexibility.
Large language models, like those developed by xAI, can generate human-like text and answer complex questions.
Climate change is a global challenge that requires immediate action to reduce carbon emissions.
RAG systems combine information retrieval and text generation to improve AI model responses.
__________________________________________________________
Generating response with LLM...

Query: What is artificial intelligence?
Response: <think>
Okay, I need to figure out what the user is asking based on their query and context. The query is "What is artificial intelligence?" and the context provided includes several sentences about AI, Python programming, language models, climate change, and RAG systems.

First, I should identify which parts of the context directly relate to defining AI. The first sentence mentions that AI is transforming the world with advances in machine learning and natural language processing (NLP). That seems relevant because it gives a functional definition of AI, tying it to specific technologies.

The other sentences mention Python programming's role in data science and AI, which is more about tools used in AI rather than defining what AI is. The part about large language models like xAI generating human-like text is related but focuses on applications of AI rather than the definition itself. Climate change isn't directly relevant here, and RAG systems are a specific type of AI system but again not a direct definition.

So, focusing on the first sentence, it's clear that AI is being discussed in terms of its technologies—machine learning and NLP. Therefore, to answer the query accurately, I should explain that AI involves these technologies, which enable tasks like generating text or understanding language through systems trained on vast data.

I need to make sure my answer is concise and directly addresses what AI is, using the provided context without adding external knowledge. It's important to connect AI to its applications but keep the definition grounded in machine learning and NLP as per the given information.
</think>

Artificial intelligence (AI) refers to technologies like machine learning and natural language processing that enable systems to perform tasks such as generating human-like text or understanding language by being trained on vast amounts of data.
"""