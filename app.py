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


