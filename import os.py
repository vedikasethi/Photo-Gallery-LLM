# Automatic Image Captioning for Gallery Images
# Using Hugging Face: nlpconnect/vit-gpt2-image-captioning

import os
import json
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# --- Settings ---
image_folder = r"C:\Users\vedik\Desktop\DLProject\iCloudPhotos"      # Folder containing your images
output_file = "image_captions.json"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load model and processor ---
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
processor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Generation parameters
max_length = 50
num_beams = 4

# --- Function to generate caption for a single image ---
def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert("RGB")  # Ensure RGB
    except Exception as e:
        print(f"Skipping {image_path}, error: {e}")
        return None

    # Preprocess image
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    # Generate caption
    with torch.no_grad():
        output_ids = model.generate(pixel_values, max_length=max_length, num_beams=num_beams)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# --- Loop over all images in folder ---
captions_dict = {}
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(image_folder, filename)
        caption = generate_caption(image_path)
        if caption:
            captions_dict[filename] = caption
            print(f"{filename} -> {caption}")

# --- Save captions to JSON ---
with open(output_file, "w") as f:
    json.dump(captions_dict, f, indent=4)

print(f"\nCaptions saved to {output_file}")
