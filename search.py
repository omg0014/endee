import os
import json
import requests
import msgpack
import google.generativeai as genai
from dotenv import load_dotenv

# Load secrets
load_dotenv()

# Configuration
ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080").rstrip("/")
if ENDEE_URL.endswith("/api/v1"):
    ENDEE_URL = ENDEE_URL[:-7]

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INDEX_NAME = "gemini_semantic_search_v3"

HEADERS = {
    "Content-Type": "application/json",
    "Bypass-Tunnel-Reminder": "true"
}

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Error: GEMINI_API_KEY not found in .env. Please add it to your .env file.")
    exit(1)

def get_embedding(text):
    """Retrieve embedding from Gemini"""
    response = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return response['embedding']

def format_result(meta_str):
    try:
        # We stored the metadata as a JSON string
        meta = json.loads(meta_str)
        if meta.get("type") == "pdf":
            return f"Found in PDF document: {meta['file']}\nRelevant chunk:\n\"{meta['content']}\""
        elif meta.get("type") == "image":
            return f"Found matching Image: {meta['file']}\nImage Description:\n\"{meta['content']}\""
        else:
            return f"Found content: {meta.get('content')}"
    except json.JSONDecodeError:
        # Fallback if it's not JSON
        return f"Relevant content:\n\"{meta_str}\""

def search_endee(query_text, k=2):
    """Embed the query and search the unified Endee index"""
    query_embedding = get_embedding(query_text)
    
    url = f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/search"
    payload = {
        "k": k,
        "vector": query_embedding
    }
    
    try:
        response = requests.post(url, json=payload, headers=HEADERS)
        
        if response.status_code == 200:
            unpacked_data = msgpack.unpackb(response.content, raw=False)
            
            if not unpacked_data or len(unpacked_data) == 0:
                print("\nNo results found.")
                return
                
            print("\n" + "="*50)
            print(f"RESULTS FOR: '{query_text}'")
            print("="*50)
            
            for i, result in enumerate(unpacked_data):
                # Endee MsgPack format: [distance, id, meta, filter, score2, vector]
                meta_data = result[2]
                
                if isinstance(meta_data, bytes):
                    meta_str = meta_data.decode('utf-8')
                else:
                    meta_str = meta_data
                    
                score = result[0] if len(result) > 0 else 0.0
                
                print(f"\n--- Result {i+1} (Score: {score:.4f}) ---")
                print(format_result(meta_str))
                
            print("\n" + "="*50)
            
        else:
            print(f"Search failed. Status: {response.status_code}, Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to Endee at {ENDEE_URL}.")
        print("Please ensure the Endee server is running locally.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    print("\n" + "*"*60)
    print(" AI Semantic Search for PDFs & Images (Endee Vector DB)")
    print("*"*60)
    print("\nType 'exit' or 'quit' to stop.")
    
    while True:
        try:
            user_query = input("\nEnter your search query: ")
            
            if user_query.lower() in ['exit', 'quit']:
                print("Exiting search...")
                break
                
            if not user_query.strip():
                continue
                
            search_endee(user_query)
            
        except KeyboardInterrupt:
            print("\nExiting search...")
            break
