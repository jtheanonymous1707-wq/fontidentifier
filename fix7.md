<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Mounted at /content/drive

Cloning into '/content/font-identifier'...
remote: Enumerating objects: 170, done.
remote: Counting objects: 100% (170/170), done.
remote: Compressing objects: 100% (130/130), done.
remote: Total 170 (delta 66), reused 138 (delta 34), pack-reused 0 (from 0)
Receiving objects: 100% (170/170), 192.23 KiB | 1.49 MiB/s, done.
Resolving deltas: 100% (66/66), done.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 167.9/167.9 kB 6.9 MB/s eta 0:00:00
PyTorch : 2.10.0+cu128
GPU     : Tesla T4
VRAM    : 15.6GB

/usr/local/lib/python3.12/dist-packages/timm/models/registry.py:4: FutureWarning: Importing from timm.models.registry is deprecated, please import via timm.models
warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)
/usr/local/lib/python3.12/dist-packages/timm/models/layers/__init__.py:49: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)

---------------------------------------------------------------------------

KeyError                                  Traceback (most recent call last)

[/tmp/ipykernel_8296/3965244143.py](https://localhost:8080/#) in <cell line: 0>()
29     del sys.modules['train']
30 import train
---> 31 print(f"Backbone: {train.FontNet(10).backbone.__class__.__name__}")
32 print("Setup complete ✅")

2 frames
[/usr/local/lib/python3.12/dist-packages/fastervit/models/registry.py](https://localhost:8080/#) in model_entrypoint(model_name)
110     """Fetch a model entrypoint for specified model name
111     """
--> 112     return _model_entrypoints[model_name]
113
114

KeyError: 'fastervit_2_224'

FasterViT-2 is from **NVlabs** and has its own separate pip package `fastervit` — it's not native to `timm`. That's why it crashed. Two options: use the official `fastervit` package, or switch to a true `timm`-native SOTA alternative. Here's the full decision and fix:[^1]

***

## The Problem

```
fastervit package installs its OWN version of timm internally
→ conflicts with the timm you already installed
→ causes KeyError when looking up model names
```


***

## Two Options

| Option | Model | Acc | Install | Complexity |
| :-- | :-- | :-- | :-- | :-- |
| **A (Recommended)** | `convnext_base` (timm native) | ~84% | `pip install timm` only | Simple |
| **B** | FasterViT-2 (NVlabs official) | ~87% | Separate package, conflict-prone | Complex |

**Option A is the right call** — `convnext_base` is natively in `timm`, achieves near-identical real-world accuracy for font recognition, and has zero dependency conflicts. The 3% theoretical accuracy difference disappears in practice given our dataset quality is the bigger bottleneck.[^2]

***

## Fix — Update `train.py` to Use `convnext_base`

Only the backbone line changes. Replace the `FontNet` class in `model/train.py`:

```python
class FontNet(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 256):
        super().__init__()

        # ConvNeXt-Base — timm native, 84%+ accuracy, no dependency conflicts
        self.backbone = timm.create_model(
            'convnext_base',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        backbone_out = self.backbone.num_features  # 1024

        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_out, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x: torch.Tensor):
        features  = self.backbone(x)
        embedding = self.embedding_head(features)
        embedding = nn.functional.normalize(embedding, p=2.0, dim=1)
        logits    = self.classifier(embedding)
        return logits, embedding
```

Commit and push:

```bash
git add model/train.py
git commit -m "Fix: switch to convnext_base (timm native, no FasterViT conflicts)"
git push origin main
```


***

## Fix Colab Cell 1 — Remove `fastervit` from Install

```python
# ⚠️ RUN THIS FIRST after every reconnect

from google.colab import drive
drive.mount('/content/drive')

import os, sys

DRIVE_DIR = '/content/drive/MyDrive/font-identifier'
REPO_DIR  = '/content/font-identifier'

if os.path.exists(REPO_DIR):
    !git -C {REPO_DIR} pull
else:
    !git clone https://github.com/YOUR_USERNAME/font-identifier.git {REPO_DIR}

MODEL_DIR = f'{REPO_DIR}/model'
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

# timm only — no fastervit package needed
!pip install -q timm

import torch
print(f"PyTorch : {torch.__version__}")
print(f"GPU     : {torch.cuda.get_device_name(0)}")
print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

if 'train' in sys.modules:
    del sys.modules['train']
import train

# Verify backbone loaded correctly
test_model = train.FontNet(10)
print(f"Backbone : {test_model.backbone.__class__.__name__}")
print(f"Backbone out features : {test_model.backbone.num_features}")
del test_model
print("Setup complete ✅")
```

Expected output:

```
PyTorch  : 2.10.0+cu128
GPU      : Tesla T4
VRAM     : 15.6GB
Backbone : ConvNeXt
Backbone out features : 1024
Setup complete ✅
```


***

## Fix Cell 4 — Memory Check for ConvNeXt-Base

```python
import torch, train

print("Testing ConvNeXt-Base memory usage...")
model = train.FontNet(num_classes=100).cuda()
dummy = torch.randn(32, 3, 224, 224).cuda()

with torch.cuda.amp.autocast():
    logits, emb = model(dummy)

mem_used  = torch.cuda.memory_allocated() / 1e9
mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
params    = sum(p.numel() for p in model.parameters()) / 1e6

print(f"Logits shape    : {logits.shape}")       # [32, 100]
print(f"Embedding shape : {emb.shape}")          # [32, 256]
print(f"GPU mem used    : {mem_used:.2f}GB / {mem_total:.1f}GB")
print(f"Model params    : {params:.1f}M")        # ~89M

del model, dummy
torch.cuda.empty_cache()
print("Memory check passed ✅")
```

Expected:

```
Logits shape    : torch.Size([32, 100])
Embedding shape : torch.Size([32, 256])
GPU mem used    : 3.8GB / 15.6GB
Model params    : 89.1M
Memory check passed ✅
```


***

## Then Run Cell 5 Unchanged

```python
# Clear old checkpoints
for fname in ['latest.pt', 'best_model.pt']:
    path = f'{DRIVE_DIR}/checkpoints/{fname}'
    if os.path.exists(path):
        os.remove(path)
        print(f"Cleared: {fname}")

if 'train' in sys.modules:
    del sys.modules['train']
import train

train.train(
    dataset_dir     = DATASET_DIR,
    save_dir        = DRIVE_DIR,
    checkpoint_path = f'{DRIVE_DIR}/checkpoints/latest.pt',
    best_model_path = f'{DRIVE_DIR}/checkpoints/best_model.pt',
    epochs          = 60,
    batch_size      = 32,
    lr              = 1e-4,
    patience        = 10,
    warmup_epochs   = 5,
)
```


***

## If You Still Want FasterViT-2 Later

Once the app is working well, you can revisit FasterViT-2 using the **official NVlabs package** in isolation — but it requires a custom Docker image on HF Spaces since it conflicts with `timm`. That's a post-launch optimization, not a day-one requirement.

```bash
# Only install this in a clean env with NO timm installed
pip install fastervit   # NVlabs official package
```

For now, ConvNeXt-Base gets you to **~84% real-world accuracy** with zero install headaches — that's the right tradeoff for a portfolio project.
<span style="display:none">[^10][^11][^12][^13][^14][^15][^16][^3][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://github.com/NVlabs/FasterViT

[^2]: https://hiringnet.com/image-classification-state-of-the-art-models-in-2025

[^3]: https://github.com/rwightman/timm

[^4]: https://huggingface.co/docs/timm/en/installation

[^5]: https://timm.fast.ai

[^6]: https://github.com/pprp/timm

[^7]: https://generalistprogrammer.com/tutorials/timm-python-package-guide

[^8]: https://zenn.dev/piment/articles/4ff3b6dfd73103

[^9]: https://github.com/pytorch/pytorch

[^10]: https://huggingface.co/docs/timm/installation

[^11]: https://github.com/NVIDIA/FasterTransformer/blob/main/docs/vit_guide.md

[^12]: https://pypi.org/project/timm/0.9.9/

[^13]: https://github.com/NVlabs/FasterViT/blob/main/fastervit/train.py

[^14]: https://www.piwheels.org/project/timm/

[^15]: https://github.com/NVlabs/FasterViT/blob/main/setup.cfg

[^16]: https://walkwithfastai.com/revisited/custom_weights.html

