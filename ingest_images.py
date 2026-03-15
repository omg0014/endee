import os
import json
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer

# Endee configuration
ENDEE_URL = "http://localhost:8080"
INDEX_NAME = "semantic_search"
DIMENSION = 384  # Dimension for 'all-MiniLM-L6-v2'
SPACE_TYPE = "cosine"
IMAGE_DIR = "data/images"

# Initialize image captioning model (BLIP)
print("Loading BLIP model for image captioning...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize embedding model (same one used for text to map to same space)
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

def generate_caption(image_path):
    """Generate a descriptive caption for an image"""
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def process_images():
    if not os.path.exists(IMAGE_DIR):
        print(f"Directory {IMAGE_DIR} not found.")
        return

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print("No Images found to ingest.")
        return

    print(f"Found {len(image_files)} image(s).")

    all_vectors = []
    vector_id = 0

    for image_file in image_files:
        image_path = os.path.join(IMAGE_DIR, image_file)
        print(f"Processing {image_file}...")
        
        # 1. Image -> Text
        caption = generate_caption(image_path)
        print(f"  Generated caption: '{caption}'")
        
        # 2. Text -> Embedding
        embedding = model.encode(caption)
        
        vector_id += 1
        meta_dict = {"type": "image", "file": image_file, "content": caption}
        all_vectors.append({
            "id": f"img_{vector_id}_{image_file}", 
            "vector": embedding.tolist(),
            "meta": json.dumps(meta_dict)
        })
            
    if all_vectors:
        print(f"Inserting {len(all_vectors)} image vectors into Endee...")
        url = f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/vector/insert"
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(url, json=all_vectors, headers=headers)
        
        if response.status_code == 200:
            print("Successfully ingested Image vectors into Endee!")
        else:
            print(f"Failed to insert vectors. Status: {response.status_code}, Error: {response.text}")

if __name__ == "__main__":
    create_index_if_not_exists()
    process_images()
