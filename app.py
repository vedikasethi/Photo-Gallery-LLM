import os
from flask import Flask, request, jsonify, send_from_directory
import torch
from PIL import Image
import io
import torch.nn.functional as F
import clip
from transformers import BlipProcessor, BlipModel, SiglipProcessor, SiglipModel

MODEL_CHOICE = None
SEARCH_MODEL = None
SEARCH_PROCESSOR = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_selected_model():
    global MODEL_CHOICE, SEARCH_MODEL, SEARCH_PROCESSOR

    while True:
        print("\n--- Please select a model to run ---")
        print("1: CLIP (ViT-B/32)")
        print("2: BLIP (Salesforce/blip-itm-base-coco)")
        print("3: SigLIP (google/siglip-base-patch16-224)")
        print("4: (Placeholder for a fourth model)")
        
        choice = input("Enter your choice (1, 2, 3, or 4): ")
        
        if choice in ['1', '2', '3', '4']:
            MODEL_CHOICE = choice
            break
        else:
            print("Invalid input. Please try again.")

    print(f"\nLoading Model {MODEL_CHOICE}...")
    if MODEL_CHOICE == '1':
        SEARCH_MODEL, SEARCH_PROCESSOR = clip.load("ViT-B/32", device=DEVICE)
        print("CLIP model loaded successfully.")
    elif MODEL_CHOICE == '2':
        model_name = "Salesforce/blip-itm-base-coco"
        SEARCH_PROCESSOR = BlipProcessor.from_pretrained(model_name)
        SEARCH_MODEL = BlipModel.from_pretrained(model_name).to(DEVICE)
        print("BLIP model loaded successfully.")
    elif MODEL_CHOICE == '3':
        model_name = "google/siglip-base-patch16-224"
        SEARCH_PROCESSOR = SiglipProcessor.from_pretrained(model_name)
        SEARCH_MODEL = SiglipModel.from_pretrained(model_name).to(DEVICE)
        print("SigLIP model loaded successfully.")
    elif MODEL_CHOICE == '4':
        print("Placeholder: No model loaded. You can add your fourth model logic here.")
        exit()

# These functions will call the correct method based on the loaded model
def encode_image(pil_image):
    with torch.no_grad():
        if MODEL_CHOICE == '1': # CLIP
            processed = SEARCH_PROCESSOR(pil_image).unsqueeze(0).to(DEVICE)
            return SEARCH_MODEL.encode_image(processed)
        else: # BLIP and SigLIP (Hugging Face)
            processed = SEARCH_PROCESSOR(images=pil_image, return_tensors="pt").to(DEVICE)
            return SEARCH_MODEL.get_image_features(**processed)

def encode_text(text_query):
    with torch.no_grad():
        if MODEL_CHOICE == '1': # CLIP
            tokens = clip.tokenize([text_query]).to(DEVICE)
            return SEARCH_MODEL.encode_text(tokens)
        else: # BLIP and SigLIP (Hugging Face)
            processed = SEARCH_PROCESSOR(text=[text_query], return_tensors="pt").to(DEVICE)
            return SEARCH_MODEL.get_text_features(**processed)

# --- Main Application Setup ---
load_selected_model()

app = Flask(__name__)
UPLOAD_FOLDER = "user_gallery"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-Memory Index
image_embeddings = {}
image_paths = {}
print("In-memory index initialized.")

# API call
@app.route('/index', methods=['POST'])
def index_image_endpoint():
    if 'image' not in request.files: return jsonify({'error': 'No image provided'}), 400
    asset_id = request.form.get('id')
    if not asset_id: return jsonify({'error': 'Asset ID is required'}), 400
    if asset_id in image_embeddings: return jsonify({'status': 'already_indexed', 'id': asset_id})

    image_file = request.files['image']
    saved_path = os.path.join(UPLOAD_FOLDER, f"{asset_id}.jpg")
    image_file.save(saved_path)

    try:
        image = Image.open(saved_path).convert("RGB")
        emb = encode_image(image) # Use the unified function
        emb /= emb.norm(dim=-1, keepdim=True)

        image_embeddings[asset_id] = emb
        image_paths[asset_id] = saved_path
        
        print(f"Indexed: {asset_id} ({len(image_embeddings)} total) using Model {MODEL_CHOICE}")
        return jsonify({'status': 'indexed', 'id': asset_id})
    except Exception as e:
        print(f"Error indexing {asset_id}: {e}")
        return jsonify({'error': f'Failed to process image: {e}'}), 500

@app.route('/search', methods=['GET'])
def search_images_endpoint():
    query = request.args.get('query')
    if not query: return jsonify({'error': 'Query parameter is required'}), 400
    if not image_embeddings: return jsonify({'results': []})

    text_emb = encode_text(query) # Use the unified function
    text_emb /= text_emb.norm(dim=-1, keepdim=True)

    ids = list(image_embeddings.keys())
    embeddings_tensor = torch.cat(list(image_embeddings.values()))

    similarities = F.cosine_similarity(text_emb, embeddings_tensor)
    
    top_k = min(20, len(ids))
    topk_results = similarities.topk(top_k)

    results_ids = [ids[i] for i in topk_results.indices.tolist()]
    return jsonify({'results': results_ids})

@app.route('/clear', methods=['POST'])
def clear_index_endpoint():
    global image_embeddings, image_paths
    image_embeddings.clear()
    image_paths.clear()
    for filename in os.listdir(UPLOAD_FOLDER):
        os.unlink(os.path.join(UPLOAD_FOLDER, filename))
    print("Index and user gallery cleared.")
    return jsonify({'status': 'cleared'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# --- Run the App --- AND BE SURE TO START "NGROK HTTP 5000"
if __name__ == '__main__':
    print(f"Starting Flask server on http://0.0.0.0:5000 with Model {MODEL_CHOICE}")
    app.run(host='0.0.0.0', port=5000) 