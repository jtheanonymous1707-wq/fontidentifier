import os, json, torch
import numpy as np

MODEL_PATH      = "font_model_scripted.pt"
CLASS_INDEX_PATH = "class_index.json"

# Load once at module import — not on every request
print("Loading font model...")
# Allow it to fail gracefully if the file isn't there yet during local dev
_model = None
if os.path.exists(MODEL_PATH):
    _model = torch.jit.load(MODEL_PATH, map_location="cpu")
    _model.eval()

_idx_to_class = {}
if os.path.exists(CLASS_INDEX_PATH):
    with open(CLASS_INDEX_PATH) as f:
        _idx_to_class = json.load(f)   # {"0": "Roboto", "1": "Open Sans", ...}

if _model:
    print(f"Model loaded. {len(_idx_to_class)} font classes.")
else:
    print("WARNING: font_model_scripted.pt not found. Inference will fail.")

def predict(tensor: "torch.Tensor") -> dict:
    """
    Returns top-5 predicted fonts with confidence scores + the 256-dim embedding.
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Ensure font_model_scripted.pt exists.")

    with torch.no_grad():
        logits, embedding = _model(tensor)

    # Top-5 predictions
    probs      = torch.softmax(logits, dim=1).squeeze()
    top5_probs, top5_idx = torch.topk(probs, k=5)

    top5 = []
    for prob, idx in zip(top5_probs, top5_idx):
        idx_str = str(idx.item())
        font_name = _idx_to_class.get(idx_str, f"Unknown ({idx_str})")
        top5.append({
            "font_name": font_name,
            "confidence": round(prob.item() * 100, 2)
        })

    return {
        "top5": top5,
        "embedding": embedding.squeeze().tolist()   # 256-dim list
    }
