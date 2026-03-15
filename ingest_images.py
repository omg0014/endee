import google.generativeai as genai
from dotenv import load_dotenv

# Load secrets
load_dotenv()

# Configuration
ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INDEX_NAME = "gemini_semantic_search_v2"
DIMENSION = 3072
SPACE_TYPE = "cosine"
IMAGE_DIR = "data/images"

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

def analyze_image_with_gemini(image_path):
    """Generate a caption using Gemini"""
    img = Image.open(image_path)
    model = genai.GenerativeModel('models/gemini-2.5-flash')
    response = model.generate_content(["Describe this image in detail for search.", img])
    return response.text

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
        response = requests.post(url, json=payload, headers=HEADERS)
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
    return analyze_image_with_gemini(image_path)

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
        embedding = get_embedding(caption)
        
        vector_id += 1
        meta_dict = {"type": "image", "file": image_file, "content": caption}
        all_vectors.append({
            "id": f"img_{vector_id}_{image_file}", 
            "vector": embedding,
            "meta": json.dumps(meta_dict)
        })
            
    if all_vectors:
        print(f"Inserting {len(all_vectors)} image vectors into Endee...")
        url = f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/vector/insert"
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(url, json=all_vectors, headers=HEADERS)
        
        if response.status_code == 200:
            print("Successfully ingested Image vectors into Endee!")
        else:
            print(f"Failed to insert vectors. Status: {response.status_code}, Error: {response.text}")

if __name__ == "__main__":
    create_index_if_not_exists()
    process_images()
