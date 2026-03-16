import os
import json
import requests
import msgpack
import google.generativeai as genai
from dotenv import load_dotenv

# Load configuration
load_dotenv()

# Endee Connection
ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080").rstrip("/")
if ENDEE_URL.endswith("/api/v1"):
    ENDEE_URL = ENDEE_URL[:-7]

INDEX_NAME = "gemini_semantic_search_v3"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Local Header configuration
HEADERS = {
    "Content-Type": "application/json",
    "Bypass-Tunnel-Reminder": "true"
}

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("❌ ERROR: GEMINI_API_KEY not found in environment.")
    exit(1)

def get_embedding(text):
    """Retrieves a vector embedding from Google Gemini."""
    try:
        response = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return response['embedding']
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return None

def display_result(score, meta_str):
    """Formats and prints a single search result."""
    print(f"\n--- [Score: {score:.4f}] ---")
    try:
        meta = json.loads(meta_str)
        source_type = meta.get("type", "Unknown").upper()
        filename = meta.get("file", "Unknown")
        content = meta.get("content", "No content available")
        
        print(f"Source: {filename} [{source_type}]")
        print(f"Content: \"{content}\"")
    except json.JSONDecodeError:
        print(f"Raw Meta: {meta_str}")

def run_semantic_search(query_text, top_k=3):
    """Performs the semantic search against the Endee vector database."""
    vector = get_embedding(query_text)
    if not vector:
        return
    
    url = f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/search"
    payload = {"k": top_k, "vector": vector}
    
    try:
        response = requests.post(url, json=payload, headers=HEADERS, timeout=30)
        
        if response.status_code == 200:
            data = msgpack.unpackb(response.content, raw=False)
            
            if not data:
                print("No relevant matches found.")
                return
                
            print(f"\n🔍 Found {len(data)} relevance results for: '{query_text}'")
            print("="*60)
            
            for result in data:
                # Endee result: [distance, id, meta, ...]
                score = result[0]
                meta_raw = result[2]
                meta_str = meta_raw.decode('utf-8') if isinstance(meta_raw, bytes) else meta_raw
                display_result(score, meta_str)
                
            print("\n" + "="*60)
        else:
            print(f"❌ Search failed (Status {response.status_code}): {response.text}")
            
    except Exception as e:
        print(f"❌ Error connecting to Endee: {e}")

if __name__ == "__main__":
    print("\n" + "🤖 Gemini AI CLI Search Toolkit".center(60))
    print("-" * 60)
    print("Directly query your Endee vector database using semantic meaning.")
    
    while True:
        try:
            query = input("\nQuery (or 'exit'): ").strip()
            if query.lower() in ['exit', 'quit', '']:
                break
            run_semantic_search(query)
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")
