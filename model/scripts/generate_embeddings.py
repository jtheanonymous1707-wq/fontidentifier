import os, json, glob, torch
import numpy as np
from PIL import Image
from torchvision import transforms
from supabase import create_client
from dotenv import load_dotenv
from train import FontNet

load_dotenv()
# SAVE_DIR should match where checkpoints are stored
SAVE_DIR  = "checkpoints" 
FONTS_DIR = "data/fonts"

def get_embedding(model, transform, font_path: str) -> list:
    from PIL import ImageFont, ImageDraw
    try:
        font = ImageFont.truetype(font_path, 32)
    except:
        return None
        
    img  = Image.new("RGB", (128, 64), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 15), "Handgloves", font=font, fill=(0, 0, 0))
    
    tensor = transform(img).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = tensor.to(device)
    
    with torch.no_grad():
        emb = model(tensor, return_embedding=True)
    return emb.squeeze().cpu().numpy().tolist()

def main():
    # Load model mapping
    class_index_path = os.path.join(SAVE_DIR, "class_index.json")
    if not os.path.exists(class_index_path):
        print(f"Error: {class_index_path} not found. Run training first.")
        return
        
    with open(class_index_path) as f:
        idx_to_class = json.load(f)   # {0: "Roboto", 1: "Open Sans", ...}

    # Load best model checkpoint
    checkpoint_path = os.path.join(SAVE_DIR, "font_model_best.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FontNet(num_classes=checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((64, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

    font_paths = glob.glob(os.path.join(FONTS_DIR, "*.ttf"))
    print(f"Generating embeddings for {len(font_paths)} fonts...")

    for font_path in font_paths:
        font_name = os.path.splitext(os.path.basename(font_path))[0]
        try:
            embedding = get_embedding(model, transform, font_path)
            if embedding is None:
                print(f"✗ {font_name}: Could not load font file.")
                continue
                
            # Update the matching row in Supabase
            supabase.table("fonts") \
                .update({"embedding": embedding}) \
                .eq("name", font_name) \
                .execute()
            print(f"✓ {font_name}")
        except Exception as e:
            print(f"✗ {font_name}: {e}")

    print("All embeddings uploaded!")

if __name__ == "__main__":
    main()
