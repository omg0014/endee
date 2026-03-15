import os
import requests
import json
import msgpack
import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from PIL import Image
import tempfile

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# The user-provided API Key
GEMINI_API_KEY = "AIzaSyBK4O8z6zvBjA4TtXURDcXQumGa9UeNTHw"
genai.configure(api_key=GEMINI_API_KEY)

# Endee configuration
ENDEE_URL = "http://localhost:8080"
INDEX_NAME = "gemini_semantic_search_v2"
DIMENSION = 3072  # GEMINI gemini-embedding-001 dimension
SPACE_TYPE = "cosine"

# Layout
st.set_page_config(page_title="AI Semantic Search", page_icon="🤖", layout="wide")

# ---------------------------------------------------------
# Database Initialization
# ---------------------------------------------------------
@st.cache_resource
def init_endee(index_name, dimension):
    """Ensure our Gemini index exists in Endee."""
    url = f"{ENDEE_URL}/api/v1/index/create"
    payload = {
        "index_name": index_name,
        "dim": dimension,
        "space_type": SPACE_TYPE,
        "precision": "float32"
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return "Successfully created index."
        elif response.status_code == 409:
            return f"Index '{index_name}' already exists."
        else:
            return f"Failed to create index. Status: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return f"Error: Could not connect to Endee at {ENDEE_URL}."

init_status = init_endee(INDEX_NAME, DIMENSION)

# Main state for tracking ingested files and their vectors
if 'vector_id' not in st.session_state:
    st.session_state.vector_id = 1

if 'ingested_files' not in st.session_state:
    st.session_state.ingested_files = {} # format: {filename: [doc_id_1, doc_id_2, ...]}

# ---------------------------------------------------------
# Gemini Wrappers
# ---------------------------------------------------------
def get_embedding(text):
    """Retrieve embedding from Gemini's gemini-embedding-001 model"""
    try:
        response = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return response['embedding']
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

def analyze_image_with_gemini(image_path):
    """Generate a highly descriptive caption for the image"""
    try:
        img = Image.open(image_path)
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        response = model.generate_content([
            "Describe this image in extreme detail so that it can be searched accurately.", 
            img
        ])
        return response.text
    except Exception as e:
        st.error(f"Error generating image caption: {e}")
        return "Unknown Image"

def generate_rag_response(query, context):
    """Generate final answer using Gemini with the retrieved context"""
    try:
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        prompt = f"""You are a helpful AI Semantic Search assistant. 
Please answer the user's question based ONLY on the provided Context documents/images. 
If the context does not contain the answer, politely state that you do not know based on the uploaded files.

Context:
{context}

User Question: {query}
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error in RAG reasoning: {e}")
        return "Sorry, there was an error generating the RAG response."


# ---------------------------------------------------------
# Ingestion Logic 
# ---------------------------------------------------------
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def ingest_to_endee(embeddings, payloads, filename):
    """Insert vectors into Endee and track their IDs"""
    url = f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/vector/insert"
    vectors = []
    vector_ids = []
    
    for emb, meta in zip(embeddings, payloads):
        if not emb: continue
        
        doc_id = f"doc_{st.session_state.vector_id}"
        vectors.append({
            "id": doc_id, 
            "vector": emb,
            "meta": json.dumps(meta)
        })
        vector_ids.append(doc_id)
        st.session_state.vector_id += 1
        
    if not vectors:
        return False, "No valid embeddings to insert."
        
    try:
        response = requests.post(url, json=vectors, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            # Track the IDs for deletion later
            if filename not in st.session_state.ingested_files:
                st.session_state.ingested_files[filename] = []
            st.session_state.ingested_files[filename].extend(vector_ids)
            
            return True, f"Successfully ingested {len(vectors)} chunks into Endee!"
        else:
            return False, f"Failed to insert vectors: {response.status_code} {response.text}"
    except Exception as e:
        return False, f"Connection error: {e}"

def delete_file_from_endee(filename):
    """Delete all vectors associated with a specific file"""
    if filename not in st.session_state.ingested_files:
        return False, f"File {filename} not found in current session tracking."
        
    vector_ids = st.session_state.ingested_files[filename]
    
    successful_deletes = 0
    total = len(vector_ids)
    
    # Endee currently supports deleting one by one via properly routed DELETE HTTP methods
    for doc_id in vector_ids:
        try:
            url = f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/vector/{doc_id}/delete"
            res = requests.delete(url)
            if res.status_code == 200:
                successful_deletes += 1
        except Exception as e:
            pass
            
    # Remove from tracking even if partial failure, to prevent ghost UI elements
    del st.session_state.ingested_files[filename]
    
    if successful_deletes == total:
         return True, f"Successfully deleted {filename} ({total} vectors)."
    else:
         return True, f"Deleted {filename} (Removed {successful_deletes}/{total} vectors)."
