<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# ---------------------------------------------------------------------------

AttributeError                            Traceback (most recent call last)

[/tmp/ipykernel_4620/972554741.py](https://localhost:8080/#) in <cell line: 0>()
5 device = torch.device('cuda')
6 \# Load best checkpoint to get num_classes
----> 7 best_ckpt = torch.load(train.BEST_MODEL_PATH, map_location=device)
8 num_classes = best_ckpt['num_classes']
9

AttributeError: module 'train' has no attribute 'BEST_MODEL_PATH'

Simple fix — since we refactored `train.py` to use function parameters instead of module-level constants, `BEST_MODEL_PATH` no longer exists on the module. Just reference the path directly using `DRIVE_DIR`:

```python
import os, torch, sys
sys.path.insert(0, '/content/font-identifier/model')
import train

DRIVE_DIR       = '/content/drive/MyDrive/font-identifier'
BEST_MODEL_PATH = f'{DRIVE_DIR}/checkpoints/best_model.pt'

device = torch.device('cuda')

# Load best checkpoint
best_ckpt   = torch.load(BEST_MODEL_PATH, map_location=device)
num_classes = best_ckpt['num_classes']

print(f"Loaded best model from epoch {best_ckpt['epoch'] + 1}")
print(f"Val accuracy: {best_ckpt['val_acc']:.4f}")
print(f"Num classes:  {num_classes}")

# Rebuild model and load weights
model = train.FontNet(num_classes=num_classes).to(device)
model.load_state_dict(best_ckpt['model_state'])
model.eval()
print("Model ready for export.")
```

Then export it:

```python
# Export as TorchScript for HF Spaces inference
model_cpu = model.cpu().eval()
scripted  = torch.jit.script(model_cpu)
scripted.save(f'{DRIVE_DIR}/font_model_scripted.pt')

size_mb = os.path.getsize(f'{DRIVE_DIR}/font_model_scripted.pt') / 1e6
print(f"✅ Model exported: {size_mb:.1f} MB")
print(f"✅ Saved to: {DRIVE_DIR}/font_model_scripted.pt")
```


***

## Rule Going Forward

Since `train.py` no longer has any module-level path constants, always define paths in the **Colab notebook** using `DRIVE_DIR` and pass them explicitly. Never reference `train.ANYTHING_PATH` — only `train.FontNet` and `train.train()` are public.

```python
# ✅ Correct — define paths in notebook
BEST_MODEL_PATH = f'{DRIVE_DIR}/checkpoints/best_model.pt'

# ❌ Wrong — module no longer has these attributes
train.BEST_MODEL_PATH
train.SAVE_DIR
train.CHECKPOINT_PATH
```

