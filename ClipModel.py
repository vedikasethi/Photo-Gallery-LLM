import torch
import clip
from PIL import Image
import json
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load CLIP ---
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)

# --- Load captions generated earlier ---
with open("image_captions.json", "r") as f:
    captions_dict = json.load(f)

image_folder = r"C:\Users\adity\Downloads\Photo-Gallery-LLM\iCloudPhotos"
image_files = list(captions_dict.keys())

# --- Precompute image embeddings ---
image_embeddings = []
for filename in image_files:
    img_path = os.path.join(image_folder, filename)
    image = preprocess_clip(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model_clip.encode_image(image)
        emb /= emb.norm(dim=-1, keepdim=True)  # normalize
    image_embeddings.append(emb)
image_embeddings = torch.cat(image_embeddings)

# --- Accuracy evaluation using generated captions ---
top1_correct = 0
top5_correct = 0
total_queries = len(captions_dict)

for filename, caption in captions_dict.items():
    text_tokens = clip.tokenize([caption]).to(device)
    with torch.no_grad():
        text_emb = model_clip.encode_text(text_tokens)
        text_emb /= text_emb.norm(dim=-1, keepdim=True)
    
    similarities = F.cosine_similarity(text_emb, image_embeddings)
    topk = similarities.topk(5)
    top5_files = [image_files[i] for i in topk.indices.tolist()]
    
    print(f"\nQuery (caption): {caption}")
    print(f"Expected image: {filename}")
    print(f"Retrieved top-5: {top5_files}")
    
    if filename == top5_files[0]:
        top1_correct += 1
    if filename in top5_files:
        top5_correct += 1

# --- Compute accuracy ---
top1_acc = (top1_correct / total_queries) * 100
top5_acc = (top5_correct / total_queries) * 100

print("\n=== Accuracy Results ===")
print(f"Top-1 Accuracy: {top1_acc:.2f}%")
print(f"Top-5 Accuracy: {top5_acc:.2f}%")

# --- Plotting ---
models = ["CLIP"]
top1 = [top1_acc]
top5 = [top5_acc]

x = range(len(models))
plt.figure(figsize=(6,5))
plt.bar([p - 0.1 for p in x], top1, width=0.2, label="Top-1")
plt.bar([p + 0.1 for p in x], top5, width=0.2, label="Top-5")
plt.xticks(x, models)
plt.ylabel("Accuracy (%)")
plt.title("Image Retrieval Accuracy (Using Generated Captions)")
plt.ylim(0, 100)
plt.legend()
plt.show()
