<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Write me a full implementation with step by step guide to start the training in colab

Important heads up before you start — Colab free tier GPU sessions now **cut off at 3–6 hours randomly**, and GPU availability isn't guaranteed daily. Since training 3,500 fonts takes ~2–3 hours, you need a bulletproof **checkpoint + auto-resume system** baked in. Here's the full guide with that handled.[^1][^2]

***

## Pre-Training Checklist (Do This Locally First)

Before opening Colab, make sure these files are in your `/model` folder and pushed to GitHub:

- `train.py` ✅ (from Phase 3)
- `scripts/download_fonts.py` ✅
- `scripts/generate_dataset.py` ✅
- `requirements.txt` ✅

***

## Step 1 — Open Colab and Connect to GPU

1. Go to [colab.research.google.com](https://colab.research.google.com) → **New Notebook**
2. Top menu → **Runtime → Change runtime type**
3. Set **Hardware accelerator → T4 GPU** → Save
4. Click **Connect** (top right) — wait for the green checkmark
5. Verify GPU is assigned:

```python
!nvidia-smi
```

You should see:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI ...   Driver Version: ...   CUDA Version: 12.x                  |
| Tesla T4     ...   15109MiB (15GB VRAM)                                    |
+-----------------------------------------------------------------------------+
```

If it says "No GPU detected", go to **Runtime → Disconnect and delete runtime** → reconnect

***

## Step 2 — Mount Google Drive

This is **critical** — all files save here so nothing is lost when Colab disconnects:

```python
from google.colab import drive
drive.mount('/content/drive')

import os

# Create your project folder in Drive
DRIVE_DIR = '/content/drive/MyDrive/font-identifier'
os.makedirs(f'{DRIVE_DIR}/checkpoints', exist_ok=True)
os.makedirs(f'{DRIVE_DIR}/dataset', exist_ok=True)
os.makedirs(f'{DRIVE_DIR}/fonts', exist_ok=True)

print("Drive mounted. Project folder ready.")
```


***

## Step 3 — Clone Your Repo

```python
# Clone your GitHub repo
!git clone https://github.com/YOUR_USERNAME/font-identifier.git /content/font-identifier

# Set working directory
%cd /content/font-identifier/model

# Verify structure
!ls -la
```

Expected output:

```
train.py
scripts/
requirements.txt
Dockerfile
```


***

## Step 4 — Install Dependencies

```python
# PyTorch is pre-installed on Colab T4, just install extras
!pip install -q timm supabase python-dotenv

# Verify GPU-enabled PyTorch
import torch
print(f"PyTorch: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

Expected:

```
PyTorch: 2.x.x+cu121
GPU available: True
GPU name: Tesla T4
```


***

## Step 5 — Set Your Environment Variables

Never hardcode secrets. In Colab, use this pattern:

```python
import os

# Paste your actual values here — this cell is not saved to GitHub
os.environ['GOOGLE_FONTS_API_KEY'] = 'your-google-fonts-api-key'
os.environ['SUPABASE_URL']         = 'https://your-project.supabase.co'
os.environ['SUPABASE_SERVICE_KEY'] = 'your-service-role-key'

print("Environment variables set.")
```


***

## Step 6 — Download Fonts (Google Fonts + Font Squirrel)

This step only needs to run **once** — after that, fonts are cached in Drive.

```python
import os, requests, zipfile, io, glob

FONTS_DIR = f'{DRIVE_DIR}/fonts'
os.makedirs(FONTS_DIR, exist_ok=True)

# ── Google Fonts ─────────────────────────────────────────
def download_google_fonts():
    api_key = os.environ['GOOGLE_FONTS_API_KEY']
    res = requests.get(f"https://www.googleapis.com/webfonts/v1/webfonts?key={api_key}&sort=alpha")
    fonts = res.json()['items']
    print(f"Found {len(fonts)} Google Fonts")

    for font in fonts:
        name = font['family'].replace(' ', '_')
        save_path = os.path.join(FONTS_DIR, f"{name}.ttf")
        if os.path.exists(save_path):
            continue
        files = font.get('files', {})
        url = files.get('regular') or list(files.values())[^0]
        if not url:
            continue
        try:
            r = requests.get(url.replace('http://', 'https://'), timeout=10)
            with open(save_path, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            print(f"  Skipped {name}: {e}")

    print("Google Fonts done.")

# ── Font Squirrel ─────────────────────────────────────────
def download_fontsquirrel():
    res = requests.get("https://www.fontsquirrel.com/api/fontlist/all", timeout=30)
    fonts = res.json()
    print(f"Found {len(fonts)} Font Squirrel fonts")

    for font in fonts:
        slug = font['family_urlname']
        name = slug.replace('-', '_')
        save_path = os.path.join(FONTS_DIR, f"{name}.ttf")
        if os.path.exists(save_path):
            continue
        try:
            url = f"https://www.fontsquirrel.com/fonts/download/{slug}"
            r = requests.get(url, timeout=20)
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                # Extract only a regular weight TTF
                for fname in z.namelist():
                    fname_lower = fname.lower()
                    is_regular = any(w in fname_lower for w in ['regular', '-400', 'book', 'roman'])
                    is_font = fname_lower.endswith('.ttf') or fname_lower.endswith('.otf')
                    if is_font and (is_regular or not any(
                        w in fname_lower for w in ['bold','italic','light','thin','black','medium']
                    )):
                        with open(save_path, 'wb') as f:
                            f.write(z.read(fname))
                        break
        except Exception as e:
            pass  # Skip failed downloads silently

    print("Font Squirrel done.")

# ── Run both ──────────────────────────────────────────────
download_google_fonts()
download_fontsquirrel()

total = len(glob.glob(os.path.join(FONTS_DIR, '*.ttf')))
print(f"\nTotal fonts downloaded: {total}")
```

Expected output:

```
Found 1521 Google Fonts
Google Fonts done.
Found 13000+ Font Squirrel fonts
Font Squirrel done.

Total fonts downloaded: ~3200
```

> 💡 **If this takes too long** (>45 mins), stop it and cap Font Squirrel at 2,000 by adding `if len(downloaded) >= 2000: break` — you can always run more later.

***

## Step 7 — Generate the Training Dataset

Create a symlink so the dataset also saves to Drive (survives disconnects):

```python
DATASET_DIR = f'{DRIVE_DIR}/dataset'
os.makedirs(DATASET_DIR, exist_ok=True)

# Point the script to Drive paths
import sys
sys.path.insert(0, '/content/font-identifier/model')
```

Now run the dataset generator — **modified to resume where it left off**:

```python
import glob, random
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np

IMG_SIZE = (128, 64)
SAMPLES_PER_FONT = 12
SAMPLE_TEXTS = [
    "Handgloves", "Typography", "AaBbCc",
    "FontID", "Design", "Hello World",
    "Quick fox", "Spectrum", "ABCDE fghij", "Waltz"
]

def apply_augmentations(img, level=1):
    img = img.convert("RGB")
    if level >= 1:
        if random.random() > 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.25))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.3))
    if level >= 2:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, random.uniform(5, 20), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img = img.rotate(random.uniform(-5, 5), fillcolor=(255, 255, 255))
        img = img.resize(IMG_SIZE, Image.LANCZOS)
    return img

def render_text_image(font_path, text, font_size=32):
    try:
        font = ImageFont.truetype(font_path, font_size)
        img  = Image.new("RGB", IMG_SIZE, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), text, font=font)
        x = max(0, (IMG_SIZE[^0] - (bbox[^2] - bbox[^0])) // 2)
        y = max(0, (IMG_SIZE[^1] - (bbox[^3] - bbox[^1])) // 2)
        color = (random.randint(0, 60),) * 3
        draw.text((x, y), text, font=font, fill=color)
        return img
    except:
        return None

font_paths = glob.glob(os.path.join(FONTS_DIR, '*.ttf'))
print(f"Generating dataset for {len(font_paths)} fonts...")
skipped, generated = 0, 0

for i, font_path in enumerate(font_paths):
    font_name = os.path.splitext(os.path.basename(font_path))[^0]
    class_dir = os.path.join(DATASET_DIR, font_name)
    os.makedirs(class_dir, exist_ok=True)

    # ← Resume: skip if already fully generated
    existing = len(os.listdir(class_dir))
    if existing >= SAMPLES_PER_FONT:
        skipped += 1
        continue

    count = 0
    for text in SAMPLE_TEXTS:
        for font_size in [28, 32, 38]:
            if count >= SAMPLES_PER_FONT:
                break
            img = render_text_image(font_path, text, font_size)
            if img is None:
                continue
            for level in [0, 1, 2]:
                aug = apply_augmentations(img.copy(), level=level)
                aug.save(os.path.join(class_dir, f"{count}_aug{level}.png"))
                count += 1
                if count >= SAMPLES_PER_FONT:
                    break

    generated += 1
    if i % 100 == 0:
        print(f"  Progress: {i}/{len(font_paths)} fonts | Generated: {generated} | Skipped: {skipped}")

total_images = len(glob.glob(os.path.join(DATASET_DIR, '*/*.png')))
total_classes = len(os.listdir(DATASET_DIR))
print(f"\nDataset ready: {total_images} images across {total_classes} font classes")
```

Expected output:

```
Generating dataset for 3200 fonts...
  Progress: 100/3200 fonts...
  Progress: 200/3200 fonts...
  ...
Dataset ready: ~38,400 images across 3,200 font classes
```


***

## Step 8 — Define the Model

```python
import torch
import torch.nn as nn
import timm

class FontNet(nn.Module):
    def __init__(self, num_classes, embedding_dim=256):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        backbone_out = self.backbone.num_features  # 1280

        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_out, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x, return_embedding=False):
        features  = self.backbone(x)
        embedding = self.embedding_head(features)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        if return_embedding:
            return embedding
        return self.classifier(embedding), embedding

device = torch.device('cuda')
print(f"Using device: {device}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```


***

## Step 9 — Set Up Data Loaders

```python
import json
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

train_transform = transforms.Compose([
    transforms.Resize((64, 128)),
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((64, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load full dataset
full_dataset = datasets.ImageFolder(DATASET_DIR)
num_classes  = len(full_dataset.classes)
print(f"Classes: {num_classes} | Total images: {len(full_dataset)}")

# Save class index — needed for inference
class_index = {v: k for k, v in full_dataset.class_to_idx.items()}
with open(f'{DRIVE_DIR}/class_index.json', 'w') as f:
    json.dump(class_index, f)
print("class_index.json saved to Drive")

# 90/10 train-val split
val_size   = int(0.1 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_set, val_set = random_split(full_dataset, [train_size, val_size],
                                   generator=torch.Generator().manual_seed(42))
train_set.dataset.transform = train_transform
val_set.dataset.transform   = val_transform

train_loader = DataLoader(train_set, batch_size=64, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
```


***

## Step 10 — Initialize Model with Checkpoint Resume

This is the most important part — **auto-resumes from the last saved checkpoint** if Colab disconnects:

```python
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

CHECKPOINT_PATH = f'{DRIVE_DIR}/checkpoints/latest.pt'
BEST_MODEL_PATH = f'{DRIVE_DIR}/checkpoints/best_model.pt'
EPOCHS  = 60
LR      = 3e-4
PATIENCE = 8

model     = FontNet(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

# ── Auto-resume from checkpoint ───────────────────────────
start_epoch    = 0
best_val_acc   = 0.0
patience_count = 0

if os.path.exists(CHECKPOINT_PATH):
    print("Checkpoint found — resuming training...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    start_epoch    = ckpt['epoch'] + 1
    best_val_acc   = ckpt['best_val_acc']
    patience_count = ckpt['patience_count']
    print(f"Resumed from epoch {start_epoch} | Best val acc so far: {best_val_acc:.4f}")
else:
    print(f"Starting fresh training | {num_classes} classes | {len(full_dataset)} images")
```


***

## Step 11 — Run the Training Loop

```python
import time

for epoch in range(start_epoch, EPOCHS):
    epoch_start = time.time()

    # ── Train ──────────────────────────────────────────────
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

        # Live batch progress every 50 batches
        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} "
                  f"| Loss: {loss.item():.4f}", end='\r')

    scheduler.step()

    # ── Validate ───────────────────────────────────────────
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits, _    = model(imgs)
            val_loss    += criterion(logits, labels).item()
            val_correct += (logits.argmax(1) == labels).sum().item()
            val_total   += labels.size(0)

    train_acc = correct / total
    val_acc   = val_correct / val_total
    elapsed   = time.time() - epoch_start

    print(f"\nEpoch {epoch+1:03d}/{EPOCHS} | "
          f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
          f"LR: {scheduler.get_last_lr()[^0]:.6f} | "
          f"Time: {elapsed:.0f}s")

    # ── Save latest checkpoint every epoch (resume safety) ──
    torch.save({
        'epoch':           epoch,
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_val_acc':    best_val_acc,
        'patience_count':  patience_count,
        'num_classes':     num_classes,
        'val_acc':         val_acc
    }, CHECKPOINT_PATH)

    # ── Save best model separately ──────────────────────────
    if val_acc > best_val_acc:
        best_val_acc   = val_acc
        patience_count = 0
        torch.save({
            'epoch':       epoch,
            'model_state': model.state_dict(),
            'num_classes': num_classes,
            'val_acc':     val_acc
        }, BEST_MODEL_PATH)
        print(f"  ✅ New best model saved (val_acc={val_acc:.4f})")
    else:
        patience_count += 1
        print(f"  No improvement ({patience_count}/{PATIENCE})")
        if patience_count >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
```


### What You'll See Per Epoch:

```
Epoch 001/060 | Train: 0.1823 | Val: 0.1541 | LR: 0.000300 | Time: 142s
  No improvement (1/8)
Epoch 002/060 | Train: 0.3912 | Val: 0.3748 | LR: 0.000298 | Time: 138s
  ✅ New best model saved (val_acc=0.3748)
...
Epoch 025/060 | Train: 0.8431 | Val: 0.8209 | LR: 0.000150 | Time: 135s
  ✅ New best model saved (val_acc=0.8209)
```


***

## Step 12 — Handle Colab Disconnects

If Colab disconnects mid-training, just **re-run Steps 1–5 and jump straight to Step 10**. The checkpoint auto-resume will pick up exactly where it left off. All fonts and dataset are already in Drive.

> 💡 **Kaggle alternative:** If Colab keeps cutting you off, [kaggle.com/code](https://kaggle.com/code) gives you **30 hrs/week of free T4 GPU** with no random disconnects. Just upload your `/model` folder as a dataset, and run the same code.[^1]

***

## Step 13 — Export the Final Model

Run this after training completes:

```python
# Load the best model
best_ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
model.load_state_dict(best_ckpt['model_state'])
model.eval()

# Export as TorchScript for faster CPU inference on HF Spaces
model_cpu = model.cpu()
model_cpu.eval()

scripted = torch.jit.script(model_cpu)
scripted.save(f'{DRIVE_DIR}/font_model_scripted.pt')

# Verify size
size_mb = os.path.getsize(f'{DRIVE_DIR}/font_model_scripted.pt') / 1e6
print(f"Model exported: {size_mb:.1f} MB")  # Expected: ~18–22MB
print(f"Final val accuracy: {best_ckpt['val_acc']:.4f}")
print(f"Trained on: {best_ckpt['num_classes']} font classes")
```


***

## Step 14 — Copy Model Back to Your Repo

Download from Drive to your local machine:

1. Go to [drive.google.com](https://drive.google.com) → `font-identifier/checkpoints/`
2. Download `font_model_scripted.pt` and `class_index.json`
3. Place both in `/model/` in your local repo
4. Commit:

```bash
cd model
git add font_model_scripted.pt class_index.json
git commit -m "Phase 3: trained FontNet on 3200 fonts, val_acc=0.82"
git push origin main
```


***

## Training Session Summary

| Step | Expected Duration |
| :-- | :-- |
| Font download (Google + FS) | 30–60 mins |
| Dataset generation | 15–20 mins |
| Model training (60 epochs) | 2–3 hrs |
| Embedding generation (Phase 3B) | 15–20 mins |
| **Total** | **~4–5 hrs** |

Split across **2 Colab sessions** if needed — the checkpoint handles the resume automatically. Once `font_model_scripted.pt` is in your repo, you're ready to move to **Phase 4: FastAPI on Hugging Face Spaces**.
<span style="display:none">[^10][^11][^12][^13][^14][^15][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://www.reddit.com/r/GoogleColab/comments/1noduls/how_long_are_the_cooldowns_for_free_tier_gpu/

[^2]: https://www.reddit.com/r/GoogleColab/comments/1ammpw2/free_tier_gpu_runtime_reduced_from_6_hours_to_3/

[^3]: http://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/

[^4]: https://www.facebook.com/groups/mlban/posts/1783547088818063/

[^5]: https://stackoverflow.com/questions/62349875/how-long-does-colabs-usage-limit-lasts

[^6]: https://techcrunch.com/2022/09/29/google-colaboratory-launches-a-pay-as-you-go-option-premium-gpu-access/

[^7]: https://www.kaggle.com/code/kadircandrisolu/efficientnet-b0-pytorch-train-birdclef-25

[^8]: https://stackoverflow.com/questions/61126851/how-can-i-use-gpu-on-google-colab-after-exceeding-usage-limit

[^9]: https://opengeoai.org/timm_train/

[^10]: https://github.com/rwightman/timm

[^11]: https://drlee.io/choosing-the-right-colab-runtime-a-guide-for-data-scientists-and-analysts-57ee7b7c9638

[^12]: https://huggingface.co/timm/tf_efficientnet_b0.in1k

[^13]: https://colab.research.google.com/signup

[^14]: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientnet.py

[^15]: https://research.google.com/colaboratory/faq.html

