<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Best epoch   : 80

Val accuracy : 0.8356
Num classes  : 1933

---------------------------------------------------------------------------

RuntimeError                              Traceback (most recent call last)

[/tmp/ipykernel_6631/3323691070.py](https://localhost:8080/#) in <cell line: 0>()
21 \# Export as TorchScript for HF Spaces inference
22 model_cpu = model.cpu().eval()
---> 23 scripted  = torch.jit.script(model_cpu)
24
25 OUTPUT_PATH = f'{DRIVE_DIR}/font_model_scripted.pt'

4 frames
[/usr/local/lib/python3.12/dist-packages/torch/jit/_recursive.py](https://localhost:8080/#) in create_methods_and_properties_from_stubs(concrete_type, method_stubs, property_stubs)
462     property_rcbs = [p.resolution_callback for p in property_stubs]
463
--> 464     concrete_type._create_methods_and_properties(
465         property_defs, property_rcbs, method_defs, method_rcbs, method_defaults
466     )

RuntimeError: Can't redefine method: forward on class: __torch__.train.FontNet (of Python compilation unit at: 0x414f6530)

83.56% accuracy — excellent result! 🎉 Now let's fix the export error.

The problem is your `forward` method returns **different types** depending on `return_embedding` — sometimes a single tensor, sometimes a tuple. TorchScript can't handle inconsistent return types.

***

## Fix 1 — Update `forward` in `train.py`

Always return both outputs as a tuple. Remove the `return_embedding` branch entirely:

```python
# In train.py — replace the forward method inside FontNet

def forward(self, x):
    features  = self.backbone(x)
    embedding = self.embedding_head(features)
    embedding = nn.functional.normalize(embedding, p=2, dim=1)
    logits    = self.classifier(embedding)
    return logits, embedding   # always return both
```

Push the change to GitHub:

```bash
# On your local machine
git add model/train.py
git commit -m "Fix: always return (logits, embedding) tuple for TorchScript compatibility"
git push origin main
```


***

## Fix 2 — Re-export in Colab (No Retraining Needed)

Pull the fix and re-export. The weights are already saved — **no retraining required**:

```python
import os, sys, torch

# Pull the updated train.py
!git -C /content/font-identifier pull

# Force reimport the updated module
if 'train' in sys.modules:
    del sys.modules['train']
import train

DRIVE_DIR       = '/content/drive/MyDrive/font-identifier'
BEST_MODEL_PATH = f'{DRIVE_DIR}/checkpoints/best_model.pt'

# Reload weights into updated model
best_ckpt   = torch.load(BEST_MODEL_PATH, map_location='cpu')
num_classes = best_ckpt['num_classes']
model       = train.FontNet(num_classes=num_classes).cpu().eval()
model.load_state_dict(best_ckpt['model_state'])

# Export with TorchScript now that forward always returns a tuple
scripted    = torch.jit.script(model)
OUTPUT_PATH = f'{DRIVE_DIR}/font_model_scripted.pt'
scripted.save(OUTPUT_PATH)

size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
print(f"Model exported ✅  {size_mb:.1f} MB")
```


***

## Fix 3 — Update `predictor.py` to Match New Output

Since `forward` no longer accepts `return_embedding`, update the embedding extraction in `predictor.py`:

```python
# In model/predictor.py — update the predict function

def predict(tensor: "torch.Tensor") -> dict:
    with torch.no_grad():
        logits, embedding = _model(tensor)   # always a tuple now

    probs = torch.softmax(logits, dim=1).squeeze()
    top5_probs, top5_idx = torch.topk(probs, k=5)

    top5 = [
        {
            "font_name": _idx_to_class[str(idx.item())],
            "confidence": round(prob.item() * 100, 2)
        }
        for prob, idx in zip(top5_probs, top5_idx)
    ]

    return {
        "top5":      top5,
        "embedding": embedding.squeeze().tolist()
    }

# Separate function used by generate_embeddings.py
def get_embedding(tensor: "torch.Tensor") -> list:
    with torch.no_grad():
        _, embedding = _model(tensor)   # take second element of tuple
    return embedding.squeeze().tolist()
```


***

## What Changed Summary

| Location | Change | Why |
| :-- | :-- | :-- |
| `train.py` `forward()` | Always returns `(logits, embedding)` | TorchScript requires consistent return type |
| `predictor.py` `predict()` | Unpacks tuple directly | Matches new forward signature |
| `predictor.py` `get_embedding()` | Takes `_[1]` from tuple | Same reason |

Commit all three changes together, then re-run Cell 6 (sanity check) to confirm the exported model runs correctly before downloading.

