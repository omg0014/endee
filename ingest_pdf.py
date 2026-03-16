import os
import json
import requests
import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv

# Initialize environment
load_dotenv()

# Configuration
ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080").rstrip("/")
if ENDEE_URL.endswith("/api/v1"):
    ENDEE_URL = ENDEE_URL[:-7]

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INDEX_NAME = "gemini_semantic_search_v3"
DIMENSION = 3072
SPACE_TYPE = "cosine"
PDF_DIR = "data/pdfs"

HEADERS = {
    "Content-Type": "application/json",
    "Bypass-Tunnel-Reminder": "true"
}

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("❌ ERROR: GEMINI_API_KEY not found.")
    exit(1)

def get_embedding(text):
    """Retrieve vector embedding from Gemini."""
    try:
        response = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return response['embedding']
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        return None

def ensure_index():
    """Ensure the Endee index is initialized."""
    url = f"{ENDEE_URL}/api/v1/index/create"
    payload = {
        "index_name": INDEX_NAME,
        "dim": DIMENSION,
        "space_type": SPACE_TYPE,
        "precision": "int8"
    }
    try:
        res = requests.post(url, json=payload, headers=HEADERS, timeout=10)
        if res.status_code == 200:
            print(f"✅ Created index: {INDEX_NAME}")
        elif res.status_code == 409:
            print(f"ℹ️ Index already exists.")
    except Exception as e:
        print(f"❌ Failed to reach Endee: {e}")
        exit(1)

def chunk_text(text, stride=500):
    """Segment text into blocks for better retrieval precision."""
    words = text.split()
    return [" ".join(words[i:i+stride]) for i in range(0, len(words), stride) if words[i:i+stride]]

def batch_ingest_pdfs():
    """Process and ingest all PDFs from the data directory."""
    if not os.path.exists(PDF_DIR):
        print(f"❌ Directory not found: {PDF_DIR}")
        return

    files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
    if not files:
        print("No PDFs found.")
        return

    print(f"🚀 Processing {len(files)} document(s)...")
    all_vectors = []
    
    for filename in files:
        path = os.path.join(PDF_DIR, filename)
        try:
            doc = fitz.open(path)
            full_text = "".join([page.get_text() for page in doc])
            chunks = chunk_text(full_text)
            
            print(f"   📄 {filename}: Generating {len(chunks)} embeddings...")
            for i, chunk in enumerate(chunks):
                emb = get_embedding(chunk)
                if emb:
                    all_vectors.append({
                        "id": f"{filename}_{i}",
                        "vector": emb,
                        "meta": json.dumps({"type": "pdf", "file": filename, "content": chunk})
                    })
        except Exception as e:
            print(f"   ⚠️ Failed to process {filename}: {e}")

    if all_vectors:
        print(f"📤 Pushing {len(all_vectors)} vectors to Endee...")
        url = f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/vector/insert"
        res = requests.post(url, json=all_vectors, headers=HEADERS, timeout=60)
        if res.status_code == 200:
            print("✅ Ingestion complete!")
        else:
            print(f"❌ Ingestion failed: {res.text}")

if __name__ == "__main__":
    ensure_index()
    batch_ingest_pdfs()
