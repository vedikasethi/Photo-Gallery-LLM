import os
from flask import Flask, request, jsonify, send_from_directory
import torch
from PIL import Image
import io
import torch.nn.functional as F
import pickle

# --- Model Imports ---
import clip
from transformers import BlipProcessor, BlipModel, SiglipProcessor, SiglipModel
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# --- Global Variables ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model 1: Captioning Model (Always Loaded) ---
print("Loading captioning model (vit-gpt2-image-captioning)...")
CAPTION_MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"
caption_model = VisionEncoderDecoderModel.from_pretrained(CAPTION_MODEL_NAME).to(DEVICE)
caption_processor = ViTImageProcessor.from_pretrained(CAPTION_MODEL_NAME)
caption_tokenizer = AutoTokenizer.from_pretrained(CAPTION_MODEL_NAME)
print("Captioning model loaded successfully.")

# --- Model 2: Semantic Search Model (User's Choice) ---
SEARCH_MODEL_CHOICE = None
SEARCH_MODEL = None
SEARCH_PROCESSOR = None

def load_selected_search_model():
    """Prompts user to select a semantic model for search."""
    global SEARCH_MODEL_CHOICE, SEARCH_MODEL, SEARCH_PROCESSOR
    while True:
        print("\n--- Please select a SEMANTIC SEARCH model ---")
        print("1: CLIP (Recommended for best performance)")
        print("2: BLIP")
        print("3: SigLIP")
        choice = input("Enter your choice (1, 2, or 3): ")
        if choice in ['1', '2', '3']:
            SEARCH_MODEL_CHOICE = choice
            break
        else:
            print("Invalid input.")
    
    print(f"\nLoading Search Model {SEARCH_MODEL_CHOICE}...")
    if SEARCH_MODEL_CHOICE == '1':
        SEARCH_MODEL, SEARCH_PROCESSOR = clip.load("ViT-B/32", device=DEVICE)
    elif SEARCH_MODEL_CHOICE == '2':
        model_name = "Salesforce/blip-itm-base-coco"
        SEARCH_PROCESSOR = BlipProcessor.from_pretrained(model_name)
        SEARCH_MODEL = BlipModel.from_pretrained(model_name).to(DEVICE)
    elif SEARCH_MODEL_CHOICE == '3':
        model_name = "google/siglip-base-patch16-224"
        SEARCH_PROCESSOR = SiglipProcessor.from_pretrained(model_name)
        SEARCH_MODEL = SiglipModel.from_pretrained(model_name).to(DEVICE)
    print(f"Search model {SEARCH_MODEL_CHOICE} loaded successfully.")

# --- Unified Text Encoding Function (for the Search Model) ---
def encode_text_for_search(text_query):
    with torch.no_grad():
        if SEARCH_MODEL_CHOICE == '1': # CLIP
            tokens = clip.tokenize([text_query]).to(DEVICE)
            return SEARCH_MODEL.encode_text(tokens)
        else: # BLIP and SigLIP
            processed = SEARCH_PROCESSOR(text=[text_query], return_tensors="pt").to(DEVICE)
            return SEARCH_MODEL.get_text_features(**processed)

# --- Main Application Setup ---
load_selected_search_model()
app = Flask(__name__)
UPLOAD_FOLDER = "user_gallery"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Caching and Indexing ---
CAPTIONS_CACHE_FILE = 'captions_cache.pkl'
EMBEDDINGS_CACHE_FILE = f'embeddings_cache_model_{SEARCH_MODEL_CHOICE}.pkl'
image_captions = {}
caption_embeddings = {}

def load_caches():
    """Loads captions and corresponding embeddings from disk."""
    global image_captions, caption_embeddings
    if os.path.exists(CAPTIONS_CACHE_FILE):
        with open(CAPTIONS_CACHE_FILE, 'rb') as f:
            image_captions = pickle.load(f)
        print(f"Loaded {len(image_captions)} captions from cache.")
    
    if os.path.exists(EMBEDDINGS_CACHE_FILE):
        with open(EMBEDDINGS_CACHE_FILE, 'rb') as f:
            caption_embeddings = pickle.load(f)
        print(f"Loaded {len(caption_embeddings)} embeddings from cache for Model {SEARCH_MODEL_CHOICE}.")
    else:
        print(f"No embedding cache found for Model {SEARCH_MODEL_CHOICE}. Will generate new ones.")
        # If captions exist but embeddings don't, create embeddings for them
        for asset_id, caption in image_captions.items():
            if asset_id not in caption_embeddings:
                 emb = encode_text_for_search(caption)
                 emb /= emb.norm(dim=-1, keepdim=True)
                 caption_embeddings[asset_id] = emb

def save_caches():
    """Saves both caches to disk."""
    with open(CAPTIONS_CACHE_FILE, 'wb') as f:
        pickle.dump(image_captions, f)
    with open(EMBEDDINGS_CACHE_FILE, 'wb') as f:
        pickle.dump(caption_embeddings, f)

load_caches()

# --- API Endpoints ---

@app.route('/index', methods=['POST'])
def index_image_endpoint():
    if 'image' not in request.files: return jsonify({'error': 'No image provided'}), 400
    asset_id = request.form.get('id')
    if not asset_id: return jsonify({'error': 'Asset ID is required'}), 400
    
    if asset_id in caption_embeddings and asset_id in image_captions:
        print(f"Asset {asset_id} already fully indexed.")
        return jsonify({'status': 'already_indexed', 'id': asset_id})

    image_file = request.files['image']
    saved_path = os.path.join(UPLOAD_FOLDER, f"{asset_id}.jpg")
    image_file.save(saved_path)

    try:
        image = Image.open(saved_path).convert("RGB")
        
        # --- Step 1: Generate Caption (if not already cached) ---
        if asset_id not in image_captions:
            pixel_values = caption_processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
            with torch.no_grad():
                output_ids = caption_model.generate(pixel_values, max_length=50, num_beams=4)
                caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            image_captions[asset_id] = caption
            print(f"Generated Caption for {asset_id}: '{caption}'")
        else:
            caption = image_captions[asset_id]

        # --- Step 2: Generate Embedding for the Caption ---
        emb = encode_text_for_search(caption)
        emb /= emb.norm(dim=-1, keepdim=True)
        caption_embeddings[asset_id] = emb
        
        save_caches()
        
        return jsonify({'status': 'indexed', 'id': asset_id, 'caption': caption})

    except Exception as e:
        print(f"Error indexing {asset_id}: {e}")
        return jsonify({'error': f'Failed to process image: {e}'}), 500

@app.route('/search', methods=['GET'])
def search_images_endpoint():
    query = request.args.get('query')
    if not query: return jsonify({'error': 'Query parameter is required'}), 400
    if not caption_embeddings: return jsonify({'results': []})

    # --- Step 3: Encode the user's search query ---
    query_emb = encode_text_for_search(query)
    query_emb /= query_emb.norm(dim=-1, keepdim=True)

    ids = list(caption_embeddings.keys())
    embeddings_tensor = torch.cat(list(caption_embeddings.values()))
    
    # --- Step 4: Perform Semantic Search ---
    similarities = F.cosine_similarity(query_emb, embeddings_tensor)
    
    top_k = min(9, len(ids)) # Limit to 9 results
    topk_results = similarities.topk(top_k)

    # Return only if the similarity is above a certain threshold
    if topk_results.values[0].item() < 0.2: # You can tune this threshold
        return jsonify({'error': f'No relevant images found for "{query}"'}), 404
        
    results_ids = [ids[i] for i in topk_results.indices.tolist()]
    return jsonify({'results': results_ids})

# ... (The /clear and /uploads endpoints remain the same but now clear both caches) ...
@app.route('/clear', methods=['POST'])
def clear_index_endpoint():
    global image_captions, caption_embeddings
    image_captions.clear()
    caption_embeddings.clear()
    save_caches()
    for filename in os.listdir(UPLOAD_FOLDER):
        os.unlink(os.path.join(UPLOAD_FOLDER, filename))
    print("Caches and user gallery cleared.")
    return jsonify({'status': 'cleared'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    print(f"Starting Flask server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000)