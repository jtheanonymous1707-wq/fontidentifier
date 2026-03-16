import io
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
import torch

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Takes raw image bytes from the upload, returns a normalized tensor.
    Handles grayscale, RGBA, and RGB inputs cleanly.
    """
    img = Image.open(io.BytesIO(image_bytes))

    # Normalize color mode
    if img.mode == "RGBA":
        # Paste onto white background to remove transparency
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Auto-level: enhance contrast on low-contrast images
    img = ImageOps.autocontrast(img, cutoff=2)

    # Mild denoise for photos
    img = img.filter(ImageFilter.MedianFilter(size=1))

    return INFERENCE_TRANSFORM(img).unsqueeze(0)  # shape: [1, 3, 64, 128]
