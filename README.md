# 🤖 Gemini AI PDF Search (Endee + Gemini)

> [!IMPORTANT]
> This project has been upgraded to **Gemini 2.0 Flash** and is now specialized for **PDF Document Intelligence**.

## 📖 Project Overview & Problem Statement

### The Problem
Traditional keyword search is "literal"—it looks for exact word matches. If you search for "security," you might miss documents about "authentication" or "access control." Furthermore, searching through hundreds of pages of unstructured PDF data is slow and manual.

### The Solution: Semantic Search
This project implements a professional **Semantic Search** system. Instead of matching words, it matches **meanings**. By converting PDF text into high-dimensional vectors (embeddings) and storing them in the **Endee Vector Database**, we can retrieve relevant information based on conceptual similarity.

- **Multimodal Reasoning**: Uses Gemini 2.0 to not just find text, but reason about it.
- **Natural Language Querying**: Ask questions like "How do I reset my password?" and get the exact paragraph from your PDFs.
- **RAG (Retrieval-Augmented Generation)**: Answers are generated using the retrieved context, ensuring accuracy and citing sources.

---

## 🏗️ System Design & Technical Approach

The pipeline follows a modern **Retrieval-Augmented Generation (RAG)** architecture:

1.  **Ingestion Phase**:
    - **Extraction**: PDF text is parsed using `PyMuPDF`.
    -  **Chunking**: Text is split into overlapping windows to preserve context.
    -  **Embedding**: Chunks are sent to Gemini's `models/gemini-embedding-001` to generate 3072-dimensional vectors.
    -  **Storage**: Vectors and original text metadata are committed to **Endee**.

2.  **Retrieval & Reasoning Phase**:
    - **Query Embedding**: The user's query is converted into a vector.
    - **Vector Search**: Endee performs a `cosine` similarity scan to find the top $K$ most relevant chunks.
    - **RAG Synthesis**: The context chunks + user query are sent to **Gemini 2.0 Flash** to generate a final, human-friendly answer.

---

## 💾 How Endee is Used

**Endee** is the high-performance engine at the heart of this project. It serves as our **Vector Database**:

- **Unified Indexing**: We use a specialized index (`gemini_semantic_search_v3`) with `int8` precision for high-speed retrieval.
- **Metadata Payloads**: Unlike traditional databases that require a separate SQL store, Endee allows us to store the **original text chunks** and **filenames** directly inside the vector payload.
- **Scalability**: Endee's C++ core allows it to handle the large 3072-dimensional Gemini vectors with sub-millisecond latency.
- **Cloud Stability**: Our implementation includes automated retries and normalized URL handling to ensure stable communication between cloud-hosted Streamlit and Endee.

---

## 🚀 Setup & Execution 

### 1. Prerequisites
- Docker (to run Endee)
- Python 3.10+
- [Google AI Studio API Key](https://aistudio.google.com/app/apikey)

### 2. Install Dependencies
```bash
git clone https://github.com/omg0014/endee.git
cd endee
pip install -r requirements.txt
```

### 3. Initialize Environment
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_key_here
ENDEE_URL=http://localhost:8080
```

### 4. Start Endee
```bash
docker run -p 8080:8080 -v ./data:/data endeeio/endee-server:latest
```

### 5. Run the Application
```bash
streamlit run app.py
```

---

## ☁️ Deployment

### Streamlit Cloud
1. Push this repo to GitHub.
2. Deploy on Streamlit Cloud and add `GEMINI_API_KEY` and `ENDEE_URL` to **Secrets**.
3. Use the provided `tunnel.sh` or a [Render.com](render_deployment_guide.md) instance for the database.

### Render.com
For 24/7 public hosting of your database, follow our [Render Deployment Guide](render_deployment_guide.md).
