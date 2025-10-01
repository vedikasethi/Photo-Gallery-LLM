# --- Imports for Model 4: BLIP-2 ---
import torch
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import Blip2Processor, Blip2Model

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load BLIP-2 Model and Processor ---
model_name_blip2 = "Salesforce/blip2-opt-2.7b"
processor_blip2 = Blip2Processor.from_pretrained(model_name_blip2, use_fast=True)
model_blip2 = Blip2Model.from_pretrained(model_name_blip2, torch_dtype=torch.float16).to(device)
print("BLIP-2 model loaded successfully.")

# --- Load captions generated earlier ---
with open("image_captions.json", "r") as f:
    captions_dict = json.load(f)

image_folder = r"C:\Users\adity\Downloads\Photo-Gallery-LLM\iCloudPhotos"
image_files = list(captions_dict.keys()) 

# --- Precompute image embeddings with BLIP-2 ---
print("Computing image embeddings with BLIP-2...")
image_embeddings_blip2 = []
for filename in image_files:
    img_path = os.path.join(image_folder, filename)
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = processor_blip2(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # Get image features from vision encoder
            vision_outputs = model_blip2.vision_model(**inputs)
            emb = model_blip2.visual_projection(vision_outputs.last_hidden_state[:, 0, :])
            emb = emb.to(torch.float32)  # Convert back to float32 for consistency
            emb /= emb.norm(dim=-1, keepdim=True)  # Normalize
        image_embeddings_blip2.append(emb)
    except Exception as e:
        print(f"Skipping {filename} due to error: {e}")

# Ensure we have a tensor of all embeddings
if not image_embeddings_blip2:
    raise ValueError("No images were processed successfully.")
image_embeddings_blip2 = torch.cat(image_embeddings_blip2)
print("Image embeddings computed.")

# --- Accuracy evaluation using generated captions ---
top1_correct_blip2 = 0
top5_correct_blip2 = 0
total_queries = len(captions_dict)

for filename, caption in captions_dict.items():
    # For BLIP-2, we need to use the language model to process text
    inputs = processor_blip2(text=[caption], return_tensors="pt").to(device)
    with torch.no_grad():
        # Get text embeddings from language model
        text_outputs = model_blip2.language_model.get_input_embeddings()(inputs.input_ids)
        # Use mean pooling for text representation
        text_emb = text_outputs.mean(dim=1)
        text_emb = text_emb.to(torch.float32)  # Convert to float32
        text_emb /= text_emb.norm(dim=-1, keepdim=True) # Normalize

    # Calculate similarity
    similarities = F.cosine_similarity(text_emb, image_embeddings_blip2)
    
    # Get top 5 results
    topk = similarities.topk(5)
    top5_files = [image_files[i] for i in topk.indices.tolist()]
    
    # Check for correctness
    if filename == top5_files[0]:
        top1_correct_blip2 += 1
    if filename in top5_files:
        top5_correct_blip2 += 1

# --- Compute and print BLIP-2 accuracy ---
top1_acc_blip2 = (top1_correct_blip2 / total_queries) * 100
top5_acc_blip2 = (top5_correct_blip2 / total_queries) * 100

print("\n=== BLIP-2 Accuracy Results ===")
print(f"Top-1 Accuracy: {top1_acc_blip2:.2f}%")
print(f"Top-5 Accuracy: {top5_acc_blip2:.2f}%")