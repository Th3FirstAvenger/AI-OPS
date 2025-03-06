"""
Test script for validating RAG functionality
Run this script to verify that RAG search is working properly
"""
import os
import json
import requests
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
import ollama

# Configuration
API_URL = "http://localhost:8000"
TEST_DATASET_NAME = "test_dataset.json"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "test_collection"
EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_HOST = "http://localhost:11434"

# Inicializar el cliente de Ollama una vez
encoder = ollama.Client(host=OLLAMA_HOST).embeddings

def embed_text(text):
    """Genera embeddings para un texto dado usando Ollama."""
    try:
        response = encoder(model=EMBEDDING_MODEL, prompt=text)
        embedding = response['embedding']
        return embedding
    except Exception as e:
        print(f"❌ Error generating embedding for text '{text[:20]}...': {e}")
        return None
    
def check_qdrant_connection():
    """Check if Qdrant is running and accessible"""
    print("\n1. Testing Qdrant Connection")
    try:
        response = requests.get(f"{QDRANT_URL}/collections")
        if response.status_code == 200:
            print("✅ Qdrant connection successful")
            collections = response.json().get("result", {}).get("collections", [])
            print(f"   Found {len(collections)} collections: {[c['name'] for c in collections]}")
            return True
        else:
            print(f"❌ Qdrant returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        return False

def create_test_dataset():
    """Create a test dataset file if it doesn't exist"""
    print("\n2. Creating Test Dataset")
    datasets_dir = Path.home() / '.aiops' / 'knowledge'

    if not datasets_dir.exists():
        print(f"   Creating datasets directory: {datasets_dir}")
        datasets_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_path = datasets_dir / TEST_DATASET_NAME
    
    # Only create if it doesn't exist
    if dataset_path.exists():
        print(f"✅ Test dataset already exists: {dataset_path}")
        return dataset_path
    
    # Create a simple test dataset
    test_data = [
        {
            "title": "Nmap Stealth Scanning",
            "content": "Nmap stealth scanning uses SYN packets (-sS) to avoid completed TCP connections. Common commands include:\n- nmap -sS -T2 target\n- nmap -sS -Pn target\n- nmap -sS -A target",
            "category": "Security Tools"
        },
        {
            "title": "SQL Injection Basics",
            "content": "SQL injection is an attack where malicious SQL code is inserted into database queries. Common tests include adding ' OR 1=1 -- to input fields.",
            "category": "Web Security"
        }
    ]
    
    try:
        with open(dataset_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        print(f"✅ Created test dataset: {dataset_path}")
        return dataset_path
    except Exception as e:
        print(f"❌ Failed to create test dataset: {e}")
        return None

def upload_to_qdrant(dataset_path):
    print("\n3. Uploading Test Dataset to Qdrant")
    try:
        client = QdrantClient(QDRANT_URL)
        if not client.collection_exists(COLLECTION_NAME):
            sample_embedding = embed_text("test")
            if sample_embedding is None:
                raise ValueError("No se pudo generar un embedding de muestra.")
            vector_size = len(sample_embedding)
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE)
            )
            print(f"   Created collection: {COLLECTION_NAME} with vector size {vector_size}")
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
            documents = data['documents']  # Accede a la lista de documentos
        
        for i, doc in enumerate(documents):
            vector = embed_text(doc['content'])  # Genera embedding del contenido
            if vector is None:
                continue
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=[rest.PointStruct(id=i, vector=vector, payload=doc)]
            )
        
        print(f"✅ Uploaded {len(documents)} documents to Qdrant collection '{COLLECTION_NAME}'")
        return True
    except Exception as e:
        print(f"❌ Failed to upload to Qdrant: {e}")
        return False

def check_api_health():
    """Check if the API is running"""
    print("\n4. Testing API Connection")
    try:
        response = requests.get(f"{API_URL}/ping")
        if response.status_code == 200:
            print("✅ API connection successful")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to API: {e}")
        return False

def test_rag_search():
    """Test the RAG search functionality directly"""
    print("\n5. Testing RAG Search Tool")
    
    # Test via agent query
    print("   Testing via agent query...")
    
    # Crear una nueva sesión
    try:
        session_response = requests.post(f"{API_URL}/sessions", params={"name": "test_rag_session"})
        session_response.raise_for_status()
        sid = session_response.json().get("sid")
        print(f"   Created test session with ID: {sid}")
        
        # Realizar una consulta RAG
        query = {
            "query": "Search our collection for information about nmap stealth scanning"
        }
        
        response = requests.post(f"{API_URL}/sessions/{sid}/chat", json=query)
        if response.status_code == 200:
            print("✅ Successfully sent RAG query to agent")
            print(f"   Response content type: {response.headers.get('content-type', 'unknown')}")
            
            # Mostrar una vista previa de la respuesta
            content = response.content.decode('utf-8') if response.content else "No content"
            print(f"   Response preview: {content[:100]}...")
            return True
        else:
            print(f"❌ Failed to query agent: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Failed to test RAG search: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("RAG FUNCTIONALITY TEST SCRIPT")
    print("=" * 50)
    
    # Ejecutar las verificaciones preliminares
    qdrant_ok = check_qdrant_connection()
    dataset_path = create_test_dataset()
    
    # Subir los datos a Qdrant si las verificaciones pasan
    if qdrant_ok and dataset_path:
        upload_ok = upload_to_qdrant(dataset_path)
        if not upload_ok:
            print("\n❌ Failed to upload dataset to Qdrant. Aborting further tests.")
            return
    
    # Verificar la API
    api_ok = check_api_health()
    
    # Proceder con la prueba de RAG si todo está bien
    if qdrant_ok and dataset_path and api_ok and upload_ok:
        print("\n✅ Preliminary checks passed. Waiting a moment for the system to initialize...")
        import time
        time.sleep(5)  # Dar tiempo al sistema para inicializarse
        
        rag_ok = test_rag_search()
        if rag_ok:
            print("\n✅ RAG search test passed")
        else:
            print("\n❌ RAG search test failed")
    else:
        print("\n❌ Preliminary checks failed. Cannot test RAG search")
    
    print("\nTest complete. Check the logs for more detailed information.")

if __name__ == "__main__":
    main()