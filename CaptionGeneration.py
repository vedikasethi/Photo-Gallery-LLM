import os
from flask import Flask, request, jsonify, send_from_directory
import torch
from PIL import Image
import io
import pickle # Used for saving and loading the cache

# --- NEW: Import the captioning model components ---
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# --- Settings & Model Loading ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Load the ViT-GPT2 model for caption generation ---
print("Loading captioning model (vit-gpt2-image-captioning)...")
CAPTION_MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"
caption_model = VisionEncoderDecoderModel.from_pretrained(CAPTION_MODEL_NAME).to(DEVICE)
caption_processor = ViTImageProcessor.from_pretrained(CAPTION_MODEL_NAME)
caption_tokenizer = AutoTokenizer.from_pretrained(CAPTION_MODEL_NAME)
print("Captioning model loaded successfully.")

# --- Main Application Setup ---
app = Flask(__name__)
UPLOAD_FOLDER = "user_gallery"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Caching and Indexing ---
CAPTIONS_CACHE_FILE = 'captions_cache.pkl'
image_captions = {} # { "phone_asset_id": "a caption for the image", ... }

def load_cache():
    """Loads the caption cache from disk at startup."""
    global image_captions
    if os.path.exists(CAPTIONS_CACHE_FILE):
        with open(CAPTIONS_CACHE_FILE, 'rb') as f:
            image_captions = pickle.load(f)
        print(f"Loaded {len(image_captions)} captions from cache.")
    else:
        print("No cache file found. Starting with an empty index.")

def save_cache():
    """Saves the current caption index to disk."""
    with open(CAPTIONS_CACHE_FILE, 'wb') as f:
        pickle.dump(image_captions, f)

# --- Load the cache when the application starts ---
load_cache()

# --- API Endpoints ---

@app.route('/index', methods=['POST'])
def index_image_endpoint():
    """Receives an image, generates a caption if not in cache, and stores it."""
    if 'image' not in request.files: return jsonify({'error': 'No image provided'}), 400
    asset_id = request.form.get('id')
    if not asset_id: return jsonify({'error': 'Asset ID is required'}), 400
    
    # --- Check Cache First ---
    if asset_id in image_captions:
        print(f"Asset {asset_id} already indexed (found in cache).")
        return jsonify({'status': 'already_indexed', 'id': asset_id})

    image_file = request.files['image']
    saved_path = os.path.join(UPLOAD_FOLDER, f"{asset_id}.jpg")
    image_file.save(saved_path)

    try:
        image = Image.open(saved_path).convert("RGB")
        
        pixel_values = caption_processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
        with torch.no_grad():
            output_ids = caption_model.generate(pixel_values, max_length=50, num_beams=4)
            caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        image_captions[asset_id] = caption
        save_cache() # Save the updated cache to disk
        
        print(f"Indexed: {asset_id} -> '{caption}'")
        return jsonify({'status': 'indexed', 'id': asset_id, 'caption': caption})

    except Exception as e:
        print(f"Error indexing {asset_id}: {e}")
        return jsonify({'error': f'Failed to process image: {e}'}), 500

@app.route('/search', methods=['GET'])
def search_images_endpoint():
    """Performs a text search, limits results, and handles no matches."""
    query = request.args.get('query')
    if not query: return jsonify({'error': 'Query parameter is required'}), 400
    if not image_captions: return jsonify({'results': []})

    search_query = query.lower().strip()
    results_ids = []

    for asset_id, caption in image_captions.items():
        if search_query in caption.lower():
            results_ids.append(asset_id)
    
    # --- Handle No Matches ---
    if not results_ids:
        print(f"Search for '{query}' found 0 results.")
        # Send a specific error message the app can check for
        return jsonify({'error': f'No images found matching "{query}"'}), 404

    # --- Limit to 9 Results ---
    final_results = results_ids[:9]
    
    print(f"Search for '{query}' found {len(results_ids)} results, returning {len(final_results)}.")
    return jsonify({'results': final_results})

@app.route('/clear', methods=['POST'])
def clear_index_endpoint():
    global image_captions
    image_captions.clear()
    save_cache() # Save the empty cache
    
    for filename in os.listdir(UPLOAD_FOLDER):
        os.unlink(os.path.join(UPLOAD_FOLDER, filename))
        
    print("Index, cache, and user gallery cleared.")
    return jsonify({'status': 'cleared'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# --- Run the App ---
if __name__ == '__main__':
    print(f"Starting Flask server on http://0.0.0.0:5000 with Captioning Model")
    app.run(host='0.0.0.0', port=5000)