import io
import torch
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),   # matches FasterViT-2 training size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes))

    # Normalize color mode
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Auto-contrast enhancement
    img = ImageOps.autocontrast(img, cutoff=2)

    return INFERENCE_TRANSFORM(img).unsqueeze(0)
