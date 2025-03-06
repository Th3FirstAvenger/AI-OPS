"""
Test script for validating RAG functionality
Run this script to verify that RAG search is working properly
"""
import os
import json
import requests
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"  # Adjust if needed
TEST_DATASET_NAME = "test_dataset.json"

def check_qdrant_connection():
    """Check if Qdrant is running and accessible"""
    print("\n1. Testing Qdrant Connection")
    try:
        response = requests.get("http://localhost:6333/collections")
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
    
    datasets_dir = Path.home() / '.aiops' / 'datasets'
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

def check_api_health():
    """Check if the API is running"""
    print("\n3. Testing API Connection")
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
    print("\n4. Testing RAG Search Tool")
    
    # Test via agent query
    print("   Testing via agent query...")
    
    # First create a new session
    try:
        session_response = requests.post(f"{API_URL}/sessions", params={"name": "test_rag_session"})
        session_response.raise_for_status()
        sid = session_response.json().get("sid")
        print(f"   Created test session with ID: {sid}")
        
        # Now try a RAG query
        query = {
            "query": "Search our collection for information about nmap stealth scanning"
        }
        
        response = requests.post(f"{API_URL}/sessions/{sid}/chat", json=query)
        if response.status_code == 200:
            print("✅ Successfully sent RAG query to agent")
            print(f"   Response content type: {response.headers.get('content-type', 'unknown')}")
            
            # Try to read a bit of the streaming response
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
    
    qdrant_ok = check_qdrant_connection()
    dataset_path = create_test_dataset()
    api_ok = check_api_health()
    
    if qdrant_ok and dataset_path and api_ok:
        print("\n✅ Preliminary checks passed. Waiting a moment for the system to initialize...")
        import time
        time.sleep(5)  # Allow time for the system to initialize
        
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