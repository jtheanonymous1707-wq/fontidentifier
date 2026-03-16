import os
import json
import glob
import torch
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from torchvision import transforms
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# Configuration
MODEL_PATH = "font_model_scripted.pt"
CLASS_INDEX_PATH = "class_index.json"
FONTS_DIR = "data/fonts"

def get_embedding(model, transform, font_path: str) -> list:
    try:
        # Render "Handgloves" (common test string for typefaces)
        font = ImageFont.truetype(font_path, 32)
        img = Image.new("RGB", (128, 64), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((10, 15), "Handgloves", font=font, fill=(0, 0, 0))
        
        # Preprocess
        tensor = transform(img).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            output = model(tensor)
            # Output is a tuple/list: [probs, embedding]
            embedding = output[1]
            return embedding.squeeze().cpu().numpy().tolist()
    except Exception as e:
        # print(f"Error processing {font_path}: {e}")
        return None

def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_INDEX_PATH):
        print("Error: Model or class index not found in current directory.")
        return

    print("Loading model...")
    model = torch.jit.load(MODEL_PATH)
    model.eval()

    with open(CLASS_INDEX_PATH) as f:
        class_index = json.load(f)
    
    # Supabase setup
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("Error: SUPABASE_URL or SUPABASE_SERVICE_KEY not set.")
        return
    supabase = create_client(url, key)

    transform = transforms.Compose([
        transforms.Resize((64, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    font_paths = glob.glob(os.path.join(FONTS_DIR, "*.ttf"))
    print(f"Generating embeddings for {len(font_paths)} fonts...")

    # We only process fonts that are in the class_index (the ones the model was trained on)
    known_fonts = set(class_index.values())
    
    count = 0
    for i, font_path in enumerate(font_paths):
        font_name_raw = os.path.splitext(os.path.basename(font_path))[0]
        font_name = font_name_raw.replace("_", " ")
        
        if i % 100 == 0:
            print(f"Progress: {i}/{len(font_paths)}... (Found: {count})")
        
        # Standardize name for matching (some may have underscores/spaces)
        # The database uses 'name', we should match that exactly.
        
        embedding = get_embedding(model, transform, font_path)
        if embedding:
            try:
                # Update Supabase
                res = supabase.table("fonts") \
                    .update({"embedding": embedding}) \
                    .eq("name", font_name) \
                    .execute()
                
                if len(res.data) > 0:
                    count += 1
            except Exception as e:
                print(f"Error uploading {font_name}: {e}")

    print(f"Finished! Uploaded {count} embeddings.")

if __name__ == "__main__":
    main()
