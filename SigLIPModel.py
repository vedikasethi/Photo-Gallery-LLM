# --- Imports for Model 3: SigLIP ---
import torch
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import SiglipProcessor, SiglipModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load SigLIP Model and Processor ---
model_name_siglip = "google/siglip-base-patch16-224"
processor_siglip = SiglipProcessor.from_pretrained(model_name_siglip, use_fast=True)
model_siglip = SiglipModel.from_pretrained(model_name_siglip).to(device)
print("SigLIP model loaded successfully.")

# --- Load captions generated earlier ---
with open("image_captions.json", "r") as f:
    captions_dict = json.load(f)

image_folder = r"C:\Users\adity\Downloads\Photo-Gallery-LLM\iCloudPhotos"
image_files = list(captions_dict.keys())

# --- Precompute image embeddings with SigLIP ---
print("Computing image embeddings with SigLIP...")
image_embeddings_siglip = []
for filename in image_files:
    img_path = os.path.join(image_folder, filename)
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = processor_siglip(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            emb = model_siglip.get_image_features(**inputs)
            emb /= emb.norm(dim=-1, keepdim=True)  # Normalize
        image_embeddings_siglip.append(emb)
    except Exception as e:
        print(f"Skipping {filename} due to error: {e}")

if not image_embeddings_siglip:
    raise ValueError("No images were processed successfully.")
image_embeddings_siglip = torch.cat(image_embeddings_siglip)
print("Image embeddings computed.")


# --- Accuracy evaluation using generated captions ---
top1_correct_siglip = 0
top5_correct_siglip = 0
total_queries = len(captions_dict)

for filename, caption in captions_dict.items():
    inputs = processor_siglip(text=[caption], return_tensors="pt").to(device)
    with torch.no_grad():
        text_emb = model_siglip.get_text_features(**inputs)
        text_emb /= text_emb.norm(dim=-1, keepdim=True) # Normalize

    similarities = F.cosine_similarity(text_emb, image_embeddings_siglip)
    
    topk = similarities.topk(5)
    top5_files = [image_files[i] for i in topk.indices.tolist()]
    
    if filename == top5_files[0]:
        top1_correct_siglip += 1
    if filename in top5_files:
        top5_correct_siglip += 1

# --- Compute and print SigLIP accuracy ---
top1_acc_siglip = (top1_correct_siglip / total_queries) * 100
top5_acc_siglip = (top5_correct_siglip / total_queries) * 100

print("\n=== SigLIP Accuracy Results ===")
print(f"Top-1 Accuracy: {top1_acc_siglip:.2f}%")
print(f"Top-5 Accuracy: {top5_acc_siglip:.2f}%")