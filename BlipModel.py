# --- Imports for Model 2: BLIP ---
import torch
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import BlipProcessor, BlipModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load BLIP Model and Processor ---
# We use a model fine-tuned for Image-Text Matching (itm)
model_name_blip = "Salesforce/blip-itm-base-coco"
processor_blip = BlipProcessor.from_pretrained(model_name_blip)
model_blip = BlipModel.from_pretrained(model_name_blip).to(device)
print("BLIP model loaded successfully.")

# --- Load captions generated earlier ---
with open("image_captions.json", "r") as f:
    captions_dict = json.load(f)

image_folder = r"C:\Users\vedik\Desktop\DLProject\iCloudPhotos"
image_files = list(captions_dict.keys()) 

# --- Precompute image embeddings with BLIP ---
print("Computing image embeddings with BLIP...")
image_embeddings_blip = []
for filename in image_files:
    img_path = os.path.join(image_folder, filename)
    try:
        image = Image.open(img_path).convert("RGB")
        # Preprocess image (does not require unsqueeze like CLIP)
        inputs = processor_blip(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            emb = model_blip.get_image_features(**inputs)
            emb /= emb.norm(dim=-1, keepdim=True)  # Normalize
        image_embeddings_blip.append(emb)
    except Exception as e:
        print(f"Skipping {filename} due to error: {e}")

# Ensure we have a tensor of all embeddings
if not image_embeddings_blip:
    raise ValueError("No images were processed successfully.")
image_embeddings_blip = torch.cat(image_embeddings_blip)
print("Image embeddings computed.")

# --- Accuracy evaluation using generated captions ---
top1_correct_blip = 0
top5_correct_blip = 0
total_queries = len(captions_dict)

for filename, caption in captions_dict.items():
    # Preprocess text and get text features
    inputs = processor_blip(text=[caption], return_tensors="pt").to(device)
    with torch.no_grad():
        text_emb = model_blip.get_text_features(**inputs)
        text_emb /= text_emb.norm(dim=-1, keepdim=True) # Normalize

    # Calculate similarity
    similarities = F.cosine_similarity(text_emb, image_embeddings_blip)
    
    # Get top 5 results
    topk = similarities.topk(5)
    top5_files = [image_files[i] for i in topk.indices.tolist()]
    
    # Check for correctness
    if filename == top5_files[0]:
        top1_correct_blip += 1
    if filename in top5_files:
        top5_correct_blip += 1

# --- Compute and print BLIP accuracy ---
top1_acc_blip = (top1_correct_blip / total_queries) * 100
top5_acc_blip = (top5_correct_blip / total_queries) * 100

print("\n=== BLIP Accuracy Results ===")
print(f"Top-1 Accuracy: {top1_acc_blip:.2f}%")
print(f"Top-5 Accuracy: {top5_acc_blip:.2f}%")