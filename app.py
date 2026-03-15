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

# Load environment variables from .env
load_dotenv()

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# The user-provided API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBK4O8z6zvBjA4TtXURDcXQumGa9UeNTHw")
genai.configure(api_key=GEMINI_API_KEY)

# Endee configuration
# First check Streamlit Secrets, then environment, then default to localhost
ENDEE_URL = st.secrets.get("ENDEE_URL", os.getenv("ENDEE_URL", "http://localhost:8080"))
INDEX_NAME = "gemini_semantic_search_v2"
DIMENSION = 3072  # GEMINI gemini-embedding-001 dimension
SPACE_TYPE = "cosine"

# Common headers to bypass localtunnel interstitial and set content type
HEADERS = {
    "Content-Type": "application/json",
    "Bypass-Tunnel-Reminder": "true"
}

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
        response = requests.post(url, json=payload, headers=HEADERS)
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
        response = requests.post(url, json=vectors, headers=HEADERS)
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
            res = requests.delete(url, headers=HEADERS)
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

# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
st.title("🤖 Gemini AI RAG + Endee Vector DB")

# Connection Status Indicator
if "Error" in init_status:
    st.error(f"📡 {init_status}")
    st.warning("⚠️ **Database Offline**: If you are on Streamlit Cloud, check your `ENDEE_URL` in Secrets and ensure your local tunnel is running.")
else:
    st.success(f"📡 Connected to Endee: {ENDEE_URL}")

tab1, tab2 = st.tabs(["Search & Chat", "Upload Data"])

with tab2:
    st.markdown("### Upload Documents & Images")
    st.write("Upload PDFs, PNGs, and JPGs. They will be embedded via the **Gemini API** and stored inside **Endee**.")
    
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=['pdf', 'png', 'jpg', 'jpeg'])
    
    if st.button("Ingest into Endee", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            with st.spinner("Processing files..."):
                for uploaded_file in uploaded_files:
                    filename = uploaded_file.name
                    ext = filename.split('.')[-1].lower()
                    
                    # Save to tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                        
                    try:
                        if ext == 'pdf':
                            st.write(f"🔄 Processing PDF: {filename}...")
                            doc = fitz.open(tmp_path)
                            text = ""
                            for page in doc:
                                text += page.get_text() + "\n"
                            
                            chunks = chunk_text(text)
                            embeddings = [get_embedding(chunk) for chunk in chunks]
                            payloads = [{"type": "pdf", "file": filename, "content": chunk} for chunk in chunks]
                            
                            success, msg = ingest_to_endee(embeddings, payloads, filename)
                            
                        elif ext in ['png', 'jpg', 'jpeg']:
                            st.write(f"🔄 Processing Image: {filename}...")
                            caption = analyze_image_with_gemini(tmp_path)
                            embedding = get_embedding(caption)
                            payloads = [{"type": "image", "file": filename, "content": caption}]
                            
                            success, msg = ingest_to_endee([embedding], payloads, filename)
                            
                        if success:
                            st.success(f"✅ {filename}: {msg}")
                        else:
                            st.error(f"❌ {filename}: {msg}")
                            
                    finally:
                        os.unlink(tmp_path)
    
    st.markdown("---")
    st.markdown("### Manage Ingested Files")
    if not st.session_state.ingested_files:
        st.info("No files currently tracked in this session.")
    else:
        for fname in list(st.session_state.ingested_files.keys()):
            colA, colB = st.columns([4, 1])
            with colA:
                st.write(f"📄 **{fname}** ({len(st.session_state.ingested_files[fname])} chunks)")
            with colB:
                if st.button("Delete", key=f"del_{fname}"):
                    with st.spinner(f"Deleting {fname}..."):
                        succ, msg = delete_file_from_endee(fname)
                        if succ:
                            st.success(msg)
                            st.rerun() # Refresh the UI dynamically
                        else:
                            st.error(msg)
                            
    st.markdown("---")
    st.markdown("### Danger Zone")
    if st.button("🗑️ Clear Entire Database", type="secondary"):
        with st.spinner("Clearing database..."):
            url = f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/delete"
            try:
                res = requests.delete(url, headers=HEADERS)
                if res.status_code == 200:
                    st.success("Database cleared successfully!")
                    st.cache_resource.clear()
                    st.session_state.vector_id = 1
                    st.session_state.ingested_files = {} # Clear tracked files
                    init_endee(INDEX_NAME, DIMENSION)
                    st.rerun()
                else:
                    st.error(f"Failed to clear database: {res.text}")
            except Exception as e:
                st.error(f"Error connecting to Endee: {e}")

with tab1:
    st.markdown("### Ask a Question")
    
    query_text = st.text_input("Enter your query:", placeholder="e.g. What is the significance of the brain diagram?")
    
    if st.button("Search", type="primary"):
        if not query_text.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Searching Endee and reasoning with Gemini..."):
                query_embedding = get_embedding(query_text)
                
                if not query_embedding:
                    st.error("Failed to embed query. Check your API key.")
                else:
                    url = f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/search"
                    payload = {"k": 3, "vector": query_embedding}
                    
                    try:
                        response = requests.post(url, json=payload, headers=HEADERS)
                        if response.status_code == 200:
                            unpacked_data = msgpack.unpackb(response.content, raw=False)
                            
                            if not unpacked_data or len(unpacked_data) == 0:
                                st.info("No relevant context found in Endee.")
                            else:
                                context_builder = ""
                                citations = []
                                
                                for result in unpacked_data:
                                    meta_data = result[2]
                                    meta_str = meta_data.decode('utf-8') if isinstance(meta_data, bytes) else meta_data
                                    score = result[0] if len(result) > 0 else 0.0
                                    
                                    try:
                                        meta = json.loads(meta_str)
                                        content = meta.get('content', meta_str)
                                        filename = meta.get('file', 'Unknown')
                                        dtype = meta.get('type', 'Unknown')
                                    except:
                                        content = meta_str
                                        filename = "Unknown"
                                        dtype = "Unknown"
                                        
                                    context_builder += f"\n--- Source: {filename} ({dtype}) ---\n{content}\n"
                                    citations.append({
                                        "score": score,
                                        "file": filename,
                                        "type": dtype,
                                        "content": content
                                    })
                                
                                # Generate RAG
                                answer = generate_rag_response(query_text, context_builder)
                                
                                st.markdown("### 🧠 AI Response")
                                st.markdown(f"> {answer}")
                                
                                st.markdown("---")
                                st.markdown("### 📚 Retrieved Citations")
                                for i, cit in enumerate(citations):
                                    with st.expander(f"[{cit['score']:.4f}] {cit['file']} ({cit['type']})"):
                                        st.write(cit['content'])
                                        
                        else:
                            st.error(f"Endee Search failed: {response.text}")
                    except Exception as e:
                        st.error(f"Error connecting to Endee: {e}")
