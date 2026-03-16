import os
import requests
import json
import msgpack
import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from PIL import Image
import tempfile
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

# ---------------------------------------------------------
# Configuration & Environment
# ---------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ **GEMINI_API_KEY Missing**: Please set your API key in Streamlit Secrets or `.env`.")
    st.stop()

# Configure Google Generative AI
genai.configure(api_key=GEMINI_API_KEY)

# Endee Database Configuration
ENDEE_URL = st.secrets.get("ENDEE_URL", os.getenv("ENDEE_URL", "http://localhost:8080"))
INDEX_NAME = "gemini_semantic_search_v3"
DIMENSION = 3072  # Dimension for gemini-embedding-001
SPACE_TYPE = "cosine"

# Clean up URL for consistency
ENDEE_URL = ENDEE_URL.rstrip("/")
if ENDEE_URL.endswith("/api/v1"):
    ENDEE_URL = ENDEE_URL[:-7]

# Global Headers
HEADERS = {
    "Content-Type": "application/json",
    "Bypass-Tunnel-Reminder": "true" # Required for localtunnel stability
}

st.set_page_config(page_title="Gemini AI PDF Search", page_icon="🤖", layout="wide")

# ---------------------------------------------------------
# Database Operations
# ---------------------------------------------------------

@st.cache_resource
def init_endee(index_name, dimension):
    """
    Ensure the vector index exists in the Endee database.
    Returns a status string describing the result.
    """
    url = f"{ENDEE_URL}/api/v1/index/create"
    payload = {
        "index_name": index_name,
        "dim": dimension,
        "space_type": SPACE_TYPE,
        "precision": "int8"
    }
    try:
        response = requests.post(url, json=payload, headers=HEADERS, timeout=30)
        if response.status_code == 200:
            return "Successfully created index."
        elif response.status_code == 409:
            return f"Index '{index_name}' already exists."
        else:
            return f"Failed to create index. Status: {response.status_code}"
    except Exception as e:
        return f"Error: Could not connect to Endee server at {ENDEE_URL}."

# Initialize background connection
with st.sidebar:
    init_status = init_endee(INDEX_NAME, DIMENSION)
    if "Error" in init_status:
        st.error(init_status)

# Session management for tracking ingested files
if 'ingested_files' not in st.session_state:
    st.session_state.ingested_files = {} # {filename: [doc_id_1, doc_id_2, ...]}

# ---------------------------------------------------------
# Gemini API Interaction
# ---------------------------------------------------------

def get_embeddings_batch(texts):
    """
    Fetches embeddings for multiple text blocks in a single API call.
    Includes sanitization to prevent NaN/Inf values.
    """
    if not texts:
        return []
    
    try:
        response = genai.embed_content(
            model="models/gemini-embedding-001",
            content=texts,
            task_type="retrieval_document"
        )
        
        # Determine the correct key for the embeddings list
        raw_embeddings = response.get('embeddings', response.get('embedding', []))
        
        # Sanitize each vector to ensure they are valid floats
        def sanitize_val(x):
            return 0.0 if (x != x or x in [float('inf'), float('-inf')]) else x

        sanitized = []
        for emb in raw_embeddings:
            if isinstance(emb, list):
                sanitized.append([sanitize_val(v) for v in emb])
            else:
                sanitized.append(sanitize_val(emb))
                
        return sanitized if 'embeddings' in response else [sanitized[0]]
        
    except Exception as e:
        err_msg = str(e).lower()
        if any(kw in err_msg for kw in ["key", "leaked", "expired", "invalid", "403", "400"]):
            st.error("🚨 **API KEY ERROR**: Please check your Gemini API key settings.")
            st.info("You can generate a new key at [Google AI Studio](https://aistudio.google.com/app/apikey).")
        else:
            st.error(f"Embedding failed: {e}")
        return []

def get_embedding(text):
    """Retrieves a single embedding for a query string."""
    res = get_embeddings_batch([text])
    return res[0] if res else None

def generate_rag_response(query, context):
    """
    Uses Gemini to reason about the retrieved context and provide an answer.
    """
    try:
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        prompt = f"""You are a helpful AI assistant. 
Answer the user question based ONLY on the provided Context. 
If the information is missing, clearly state that the documents do not contain the answer.

Context:
{context}

User Question: {query}
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        err_msg = str(e).lower()
        if "quota" in err_msg:
            st.error("🚨 **QUOTA EXCEEDED**: Gemini Free Tier limit reached. Please try again later.")
        else:
            st.error(f"Reasoning error: {e}")
        return "I apologize, but I encountered an error while processing your request."

# ---------------------------------------------------------
# Document Ingestion
# ---------------------------------------------------------

def chunk_text(text, chunk_size=300):
    """Splits text into manageable chunks for vector search."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def ingest_to_endee(embeddings, payloads, filename):
    """
    Pushes vectors to Endee with automatic retry logic for transient network issues.
    """
    url = f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/vector/insert"
    vectors = []
    vector_ids = []
    
    for emb, meta in zip(embeddings, payloads):
        if not emb: continue
        
        doc_id = str(uuid.uuid4())
        vectors.append({
            "id": doc_id, 
            "vector": emb,
            "meta": json.dumps(meta)
        })
        vector_ids.append(doc_id)
        
    if not vectors:
        return False, "No valid content to ingest."
        
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=vectors, headers=HEADERS, timeout=30)
            
            if response.status_code == 200:
                if filename not in st.session_state.ingested_files:
                    st.session_state.ingested_files[filename] = []
                st.session_state.ingested_files[filename].extend(vector_ids)
                return True, f"Ingested {len(vectors)} chunks"
            
            # Handle specific metadata sync errors (Render persistent disk issues)
            if response.status_code == 400 and "Required files missing" in response.text:
                st.warning("⚠️ **Database Desync Detected**: Metadata is present but files are missing.")
                if st.button("🔨 Repair Database Now", type="primary"):
                     with st.spinner("Repairing..."):
                         requests.delete(f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/delete", headers=HEADERS, timeout=30)
                         st.cache_resource.clear()
                         init_endee(INDEX_NAME, DIMENSION)
                         st.success("Database repaired! Please try again.")
                         st.rerun()
                return False, "Database requires repair."

            return False, f"Server Error: {response.status_code}"
            
        except Exception:
            if attempt < max_retries - 1:
                continue
            return False, "Connection failure after retries."

def delete_file_from_endee(filename):
    """Removes all vectors associated with a specific file from the index."""
    if filename not in st.session_state.ingested_files:
        return False, "File not tracked."
        
    vector_ids = st.session_state.ingested_files[filename]
    success_count = 0
    
    for doc_id in vector_ids:
        try:
            url = f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/vector/{doc_id}/delete"
            res = requests.delete(url, headers=HEADERS, timeout=15)
            if res.status_code == 200:
                success_count += 1
        except Exception:
            pass
            
    del st.session_state.ingested_files[filename]
    return True, f"Removed {success_count}/{len(vector_ids)} chunks."

# ---------------------------------------------------------
# Application UI
# ---------------------------------------------------------

st.title("🤖 Gemini AI PDF Search + Endee Vector DB")

if "Error" in init_status:
    st.error(f"📡 Connection Status: {init_status}")
    st.warning("⚠️ Ensure your Endee server is reachable at the configured URL.")

tab_search, tab_upload = st.tabs(["Search & Chat", "Upload Documents"])

with tab_upload:
    st.markdown("### 📄 Upload Documents")
    st.write("Upload PDF files to build your searchable knowledge base.")
    
    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type=['pdf'])
    
    if st.button("Ingest Documents", type="primary"):
        if not uploaded_files:
            st.warning("Please select at least one PDF.")
        else:
            with st.spinner("Processing documents..."):
                for uploaded_file in uploaded_files:
                    filename = uploaded_file.name
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                        
                    try:
                        st.write(f"🔍 Analyzing: {filename}...")
                        doc = fitz.open(tmp_path)
                        text = "".join([page.get_text() + "\n" for page in doc])
                        
                        chunks = chunk_text(text)
                        if not chunks:
                            st.warning(f"⚠️ {filename} is empty.")
                            continue

                        st.write(f"🧠 Generating embeddings...")
                        embeddings = get_embeddings_batch(chunks)
                        payloads = [{"type": "pdf", "file": filename, "content": chunk} for chunk in chunks]
                        
                        success, msg = ingest_to_endee(embeddings, payloads, filename)
                        if success:
                            st.success(f"✅ {filename}: {msg}")
                        else:
                            st.error(f"❌ {filename}: {msg}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
    
    st.markdown("---")
    st.markdown("### 📋 Manage Files")
    if not st.session_state.ingested_files:
        st.info("No files in the current index yet.")
    else:
        for fname in list(st.session_state.ingested_files.keys()):
            col_name, col_del = st.columns([4, 1])
            with col_name:
                st.write(f"📄 **{fname}** ({len(st.session_state.ingested_files[fname])} chunks)")
            with col_del:
                if st.button("Delete", key=f"del_{fname}"):
                    with st.spinner("Deleting..."):
                        succ, msg = delete_file_from_endee(fname)
                        st.rerun()
                            
    st.markdown("---")
    st.markdown("### ⚠️ Danger Zone")
    if st.button("🗑️ Reset Entire Database", type="secondary", help="Irreversible: wipes all stored data."):
        with st.spinner("Clearing everything..."):
            try:
                requests.delete(f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/delete", headers=HEADERS, timeout=30)
                st.cache_resource.clear()
                st.session_state.ingested_files = {} 
                init_endee(INDEX_NAME, DIMENSION)
                st.success("Database fully reset.")
                st.rerun()
            except Exception as e:
                st.error(f"Reset failed: {e}")

with tab_search:
    st.markdown("### 🔍 Semantic Ask")
    query_text = st.text_input("Ask a question about your documents:", placeholder="e.g. What is the main conclusion of the report?")
    
    if st.button("Search", type="primary"):
        if not query_text.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                query_embedding = get_embedding(query_text)
                
                if query_embedding:
                    url = f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/search"
                    payload = {"k": 3, "vector": query_embedding}
                    
                    try:
                        response = requests.post(url, json=payload, headers=HEADERS, timeout=30)
                        if response.status_code == 200:
                            results = msgpack.unpackb(response.content, raw=False)
                            
                            if not results:
                                st.info("No matching information found.")
                            else:
                                context = ""
                                sources = []
                                
                                for r in results:
                                    # result: [distance, id, meta, ...]
                                    raw_meta = r[2]
                                    meta_str = raw_meta.decode('utf-8') if isinstance(raw_meta, bytes) else raw_meta
                                    try:
                                        meta = json.loads(meta_str)
                                        content = meta.get('content', 'No content')
                                        fname = meta.get('file', 'Unknown')
                                    except:
                                        content, fname = meta_str, "Unknown"
                                        
                                    context += f"\nSOURCE [{fname}]:\n{content}\n"
                                    sources.append({"name": fname, "score": r[0], "text": content})
                                
                                answer = generate_rag_response(query_text, context)
                                
                                st.markdown("#### 🧠 Answer")
                                st.write(answer)
                                
                                st.markdown("---")
                                st.markdown("#### 📚 Reference Chunks")
                                for s in sources:
                                    with st.expander(f"📄 {s['name']} (Similarity: {s['score']:.4f})"):
                                        st.write(s['text'])
                        else:
                            st.error("Search engine returned an error.")
                    except Exception as e:
                        st.error(f"Search failed: {e}")
