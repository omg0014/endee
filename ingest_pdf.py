import os
import fitz  # PyMuPDF
import json
import requests
from sentence_transformers import SentenceTransformer

# Endee configuration
ENDEE_URL = "http://localhost:8080"
INDEX_NAME = "semantic_search"
DIMENSION = 384  # Dimension for 'all-MiniLM-L6-v2'
SPACE_TYPE = "cosine"
PDF_DIR = "data/pdfs"

# Initialize embedding model
print("Loading sentence-transformers model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_index_if_not_exists():
    """Create the unified search index in Endee vector database"""
    print(f"Ensuring index '{INDEX_NAME}' exists...")
    url = f"{ENDEE_URL}/api/v1/index/create"
    
    payload = {
        "index_name": INDEX_NAME,
        "dim": DIMENSION,
        "space_type": SPACE_TYPE,
        "precision": "int8"
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Successfully created index.")
        elif response.status_code == 409:
            print(f"Index '{INDEX_NAME}' already exists.")
        else:
            print(f"Failed to create index. Status: {response.status_code}, Response: {response.text}")
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to Endee at {ENDEE_URL}. Make sure it is running.")
        exit(1)

def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

def chunk_text(text, chunk_size=500):
    """Splits a long string into chunks roughly of size `chunk_size` words"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def process_pdfs():
    if not os.path.exists(PDF_DIR):
        print(f"Directory {PDF_DIR} not found.")
        return

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("No PDFs found to ingest.")
        return

    print(f"Found {len(pdf_files)} PDF(s).")

    all_vectors = []
    vector_id = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        print(f"Reading {pdf_file}...")
        
        full_text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(full_text)
        
        print(f"Extracted {len(chunks)} chunks from {pdf_file}. Generating embeddings...")
        embeddings = model.encode(chunks)
        
        for chunk, embedding in zip(chunks, embeddings):
            vector_id += 1
            meta_dict = {"type": "pdf", "file": pdf_file, "content": chunk}
            all_vectors.append({
                "id": f"pdf_{vector_id}_{pdf_file}", 
                "vector": embedding.tolist(),
                "meta": json.dumps(meta_dict)
            })
            
    if all_vectors:
        print(f"Inserting {len(all_vectors)} text vectors into Endee...")
        url = f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/vector/insert"
        headers = {"Content-Type": "application/json"}
        
        # Insert in batches if too large, but for a simple project one request is fine
        response = requests.post(url, json=all_vectors, headers=headers)
        
        if response.status_code == 200:
            print("Successfully ingested PDF vectors into Endee!")
        else:
            print(f"Failed to insert vectors. Status: {response.status_code}, Error: {response.text}")

if __name__ == "__main__":
    create_index_if_not_exists()
    process_pdfs()
