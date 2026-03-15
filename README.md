# AI Semantic Search for PDFs and Images using Endee Vector Database

## Project Overview

Traditional keyword search fails to understand context and meaning—it only looks for exact word matches. **Semantic search** overcomes this by retrieving results based on the *meaning* of the queries and documents using dense vector embeddings.

This project implements a professional, clean, and beginner-friendly AI Semantic Search system. It allows users to search for information across both **PDF documents** and **Images** using natural language queries seamlessly. 

The system relies heavily on the **Endee Vector Database** for lightning-fast similarity search of the high-dimensional embeddings.

## Problem Statement

Traditional search systems break when users use synonyms or describe concepts instead of using exact keywords. Furthermore, searching across multimodal data (like texts and images simultaneously) is difficult using traditional full-text engines.

Semantic search solves this by:
1. Converting text and image content into mathematical arrays (vector embeddings) where semantically similar concepts are closer together.
2. Converting the user's natural language query into the same vector space.
3. Rapidly retrieving the closest vectors using a vector database.

## System Architecture

The pipeline consists of three separate steps: extracting and embedding PDFs, extracting and embedding images, and searching. All components map to the same vector dimension (`384`) using the `all-MiniLM-L6-v2` transformer model so they can be queried simultaneously.

```
PDF / Image Files
       ↓
Text Extraction (PyMuPDF) / Image Captioning (BLIP)
       ↓
Embedding Model (all-MiniLM-L6-v2)
       ↓
Store embeddings in Endee Vector Database
       ↓
User Query ("neural networks")
       ↓
Query Embedding (all-MiniLM-L6-v2)
       ↓
Vector Similarity Search in Endee
       ↓
Return Most Relevant Results
```

## How Endee is Used

**Endee** is an open-source, high-performance vector database. In this project:
- We create a unified space (`semantic_search`) using the `cosine` distance metric.
- Endee stores our highly dense `384-dimensional` vector embeddings.
- Endee's HTTP REST API allows us to easily create indexes, insert vectors, and perform nearest-neighbor searches at scale.
- We leverage Endee's `meta` payload capability to seamlessly store the corresponding text chunks and image captions as stringified JSON directly alongside the vectors. This prevents the need for a secondary relational database to fetch document text.

## Setup Instructions

### Step 1: Clone Repository

```bash
git clone https://github.com/omg0014/endee.git
cd endee
```

### Step 2: Install Dependencies

We recommend using a python virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Start Endee Vector Database

Ensure Docker is installed, then start Endee on `localhost:8080`:

```bash
docker run \
  --ulimit nofile=100000:100000 \
  -p 8080:8080 \
  -v ./endee-data:/data \
  --name endee-server \
  --restart unless-stopped \
  endeeio/endee-server:latest
```

*Alternatively, you can build it locally from the `endee` repository source.*

### Step 4: Add Data

1. Place any PDF files you want to search through into the `data/pdfs/` directory.
2. Place any Images (jpg, png) you want to search through into the `data/images/` directory.

### Step 5: Run Ingestion

Run the ingestion scripts to extract text/captions, generate embeddings, and store them into Endee:

```bash
python ingest_pdf.py
python ingest_images.py
```

### Step 6: Run Semantic Search

Interact with the database using natural language queries:

```bash
python search.py
```

## Example Query

**Enter query:** `"deep learning"`

**Expected output:**
```
==================================================
RESULTS FOR: 'deep learning'
==================================================

--- Result 1 (Score: 0.7412) ---
Found in PDF document: ai_notes.pdf
Relevant chunk:
"Deep learning is a subset of machine learning that uses multi-layered artificial neural networks to deliver state-of-the-art accuracy in tasks such as object detection, speech recognition, and language translation."

--- Result 2 (Score: 0.6120) ---
Found matching Image: neural_network_diagram.jpg
Image Description:
"a diagram showing a deep neural network with input, hidden, and output layers"

==================================================
```
