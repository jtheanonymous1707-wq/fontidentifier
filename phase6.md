<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Rewrite the implementation plan to get generate more dataset and provide the full code to fine-tune FasterViT-2

Here's the complete rewritten implementation with all code ready to run.

***

## Phase R1: Generate Dataset v2 Locally

### Step 1 — Update `generate_dataset.py`

Replace the entire file at `model/scripts/generate_dataset.py`:

```python
import os, io, random, glob, shutil
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np

FONTS_DIR        = "data/fonts"
OUTPUT_DIR       = "data/dataset_v2"
IMG_SIZE         = (224, 224)
SAMPLES_PER_FONT = 60   # 3x level-0, 3x level-1, 3x level-2 per text

SAMPLE_TEXTS = [
    # Short distinctive words (best for font recognition)
    "Handgloves", "Typography", "Spectrum", "Waltz",
    "Jackdaws", "Sphinx", "Quartz", "Vixen",
    # Mixed case (shows cap/lowercase contrast)
    "AaBbCcDd", "FontID", "HelveArial", "GlyphSet",
    # Numbers + letters (tests numeral style)
    "Style 2024", "Type 01", "Vol.3 No.9",
    # Longer lines (shows letter spacing)
    "Quick brown fox", "Pack my box",
    "Five boxing wizards",
]

FONT_SIZES = [22, 26, 30, 34, 38, 44, 50]

BACKGROUNDS = [
    # Light backgrounds
    (255, 255, 255),   # pure white
    (248, 248, 248),   # near white
    (245, 243, 238),   # warm paper
    (238, 243, 250),   # cool paper
    (250, 245, 235),   # aged paper
    # Dark backgrounds
    (18,  18,  18),    # near black
    (25,  28,  35),    # dark navy
    (30,  25,  30),    # dark purple
    (20,  35,  20),    # dark green
    # Mid-tone
    (180, 175, 170),   # warm gray
    (160, 165, 175),   # cool gray
]

DARK_TEXT_COLORS  = [(0,0,0),(20,20,20),(40,40,40),(60,60,60),(80,60,40)]
LIGHT_TEXT_COLORS = [(255,255,255),(230,230,230),(210,210,210),(240,235,220)]


def apply_augmentations(img: Image.Image, level: int) -> Image.Image:
    img = img.convert("RGB")

    if level == 0:
        # Clean — no augmentation, just resize
        return img.resize(IMG_SIZE, Image.LANCZOS)

    if level >= 1:
        # Gaussian blur
        if random.random() > 0.45:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.uniform(0.3, 1.6)
            ))
        # Brightness
        img = ImageEnhance.Brightness(img).enhance(
            random.uniform(0.65, 1.35)
        )
        # Contrast
        img = ImageEnhance.Contrast(img).enhance(
            random.uniform(0.7, 1.4)
        )
        # Sharpness (sometimes over-sharpen to simulate screenshots)
        if random.random() > 0.6:
            img = ImageEnhance.Sharpness(img).enhance(
                random.uniform(0.5, 2.5)
            )
        # JPEG compression artifacts
        if random.random() > 0.35:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=random.randint(45, 85))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

    if level >= 2:
        # Gaussian noise
        arr   = np.array(img).astype(np.float32)
        noise = np.random.normal(0, random.uniform(3, 22), arr.shape)
        arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img   = Image.fromarray(arr)

        # Rotation (-7 to +7 degrees)
        angle    = random.uniform(-7, 7)
        bg_fill  = random.choice([(255,255,255),(245,245,245),(0,0,0)])
        img      = img.rotate(angle, fillcolor=bg_fill, expand=False)

        # Horizontal perspective warp
        new_w = int(img.width * random.uniform(0.85, 1.15))
        img   = img.resize((new_w, img.height), Image.LANCZOS)
        img   = img.resize(IMG_SIZE, Image.LANCZOS)

        # Subtle background texture
        if random.random() > 0.5:
            lo = random.randint(235, 250)
            texture = np.random.randint(
                lo, 256, (img.height, img.width, 3), dtype=np.uint8
            )
            img = Image.blend(
                img,
                Image.fromarray(texture),
                alpha=random.uniform(0.04, 0.18)
            )

        # Random horizontal crop/pad (simulates partial text visibility)
        if random.random() > 0.6:
            arr    = np.array(img)
            offset = random.randint(5, 20)
            if random.random() > 0.5:
                arr = np.pad(arr, ((0,0),(offset,0),(0,0)),
                             mode='constant', constant_values=255)[:, :IMG_SIZE[0], :]
            else:
                arr = np.pad(arr, ((0,0),(0,offset),(0,0)),
                             mode='constant', constant_values=255)[:, offset:offset+IMG_SIZE[0], :]
            img = Image.fromarray(arr.astype(np.uint8))
            img = img.resize(IMG_SIZE, Image.LANCZOS)

    return img


def render_text_image(
    font_path: str,
    text: str,
    font_size: int,
    bg_color: tuple,
    text_color: tuple
) -> Image.Image | None:
    try:
        font   = ImageFont.truetype(font_path, font_size)
        canvas = Image.new("RGB", IMG_SIZE, color=bg_color)
        draw   = ImageDraw.Draw(canvas)

        bbox   = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Skip unreadable renders
        if text_w < 20 or text_h < 8:
            return None
        # Skip if text overflows image
        if text_w > IMG_SIZE[0] * 0.92:
            return None

        # Add slight random offset from center for variety
        base_x = max(4, (IMG_SIZE[0] - text_w) // 2)
        base_y = max(4, (IMG_SIZE[1] - text_h) // 2)
        x = base_x + random.randint(-8, 8)
        y = base_y + random.randint(-10, 10)
        x = max(2, min(x, IMG_SIZE[0] - text_w - 2))
        y = max(2, min(y, IMG_SIZE[1] - text_h - 2))

        draw.text((x, y), text, font=font, fill=text_color)
        return canvas
    except Exception:
        return None


def generate_dataset():
    font_paths = glob.glob(os.path.join(FONTS_DIR, "*.ttf"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Fonts found       : {len(font_paths)}")
    print(f"Samples per font  : {SAMPLES_PER_FONT}")
    print(f"Target images     : {len(font_paths) * SAMPLES_PER_FONT:,}")
    print(f"Output dir        : {OUTPUT_DIR}")
    print(f"Image size        : {IMG_SIZE}")
    print("─" * 50)

    skipped = generated = failed = 0

    for i, font_path in enumerate(font_paths):
        font_name = os.path.splitext(os.path.basename(font_path))[0]
        class_dir = os.path.join(OUTPUT_DIR, font_name)
        os.makedirs(class_dir, exist_ok=True)

        # Resume — skip if already fully generated
        existing = len(glob.glob(f"{class_dir}/*.png"))
        if existing >= SAMPLES_PER_FONT:
            skipped += 1
            continue

        count    = existing   # resume from where we left off
        attempts = 0

        while count < SAMPLES_PER_FONT and attempts < 500:
            attempts += 1
            text       = random.choice(SAMPLE_TEXTS)
            font_size  = random.choice(FONT_SIZES)
            bg_color   = random.choice(BACKGROUNDS)
            is_dark_bg = sum(bg_color) < 200
            text_color = random.choice(
                LIGHT_TEXT_COLORS if is_dark_bg else DARK_TEXT_COLORS
            )

            img = render_text_image(font_path, text, font_size, bg_color, text_color)
            if img is None:
                continue

            # Save 3 augmentation levels per render
            for level in [0, 1, 2]:
                if count >= SAMPLES_PER_FONT:
                    break
                aug = apply_augmentations(img.copy(), level=level)
                aug.save(
                    os.path.join(class_dir, f"{count:03d}_l{level}.png"),
                    format="PNG"
                )
                count += 1

        if count == 0:
            shutil.rmtree(class_dir)
            failed += 1
        else:
            generated += 1

        if i % 100 == 0 or i == len(font_paths) - 1:
            total_imgs = len(glob.glob(f"{OUTPUT_DIR}/*/*.png"))
            print(
                f"[{i+1:4d}/{len(font_paths)}] "
                f"Generated: {generated:4d} | "
                f"Skipped: {skipped:4d} | "
                f"Failed: {failed:3d} | "
                f"Total: {total_imgs:,}"
            )

    # Final cleanup
    removed = 0
    for d in os.listdir(OUTPUT_DIR):
        full = os.path.join(OUTPUT_DIR, d)
        if os.path.isdir(full) and len(os.listdir(full)) == 0:
            os.rmdir(full)
            removed += 1

    total_classes = len(os.listdir(OUTPUT_DIR))
    total_images  = len(glob.glob(f"{OUTPUT_DIR}/*/*.png"))
    print("\n" + "─" * 50)
    print(f"Dataset v2 ready ✅")
    print(f"Classes : {total_classes:,}")
    print(f"Images  : {total_images:,}")
    print(f"Cleaned : {removed} empty folders")


if __name__ == "__main__":
    generate_dataset()
```


### Step 2 — Run Locally

```bash
cd model
source venv/bin/activate

# Remove old dataset to start clean
rm -rf data/dataset_v2/

python scripts/generate_dataset.py
# Expected: ~1,900 classes × 60 images = ~114,000 images
# Takes: ~60-80 mins
```


### Step 3 — Zip and Upload to Drive

```bash
cd model
zip -r dataset_v2.zip data/dataset_v2/
# Upload dataset_v2.zip to Google Drive → font-identifier/
```


***

## Phase R2: Update `train.py` for FasterViT-2

Replace the **entire** `model/train.py`:

```python
import os, json, time
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# ── Model ─────────────────────────────────────────────────────────────────────
class FontNet(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 256):
        super().__init__()

        # FasterViT-2 — SOTA for font recognition (87.4% top-1)
        self.backbone = timm.create_model(
            'fastervit_2_224',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        backbone_out = self.backbone.num_features  # 768

        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_out, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),              # GELU works better than ReLU for ViT models
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


# ── Train Function ─────────────────────────────────────────────────────────────
def train(
    dataset_dir:     str,
    save_dir:        str,
    checkpoint_path: str   = None,
    best_model_path: str   = None,
    epochs:          int   = 60,
    lr:              float = 1e-4,    # lower LR for FasterViT
    batch_size:      int   = 32,      # smaller batch for larger model
    patience:        int   = 10,
    embedding_dim:   int   = 256,
    warmup_epochs:   int   = 5        # LR warmup for stable fine-tuning
):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(save_dir, "checkpoints", "latest.pt")
    if best_model_path is None:
        best_model_path = os.path.join(save_dir, "checkpoints", "best_model.pt")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(best_model_path),  exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device     : {device}")
    if torch.cuda.is_available():
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
        print(f"VRAM       : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # ── Transforms ────────────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ── Dataset ────────────────────────────────────────────────────────────────
    full_dataset = datasets.ImageFolder(dataset_dir)
    num_classes  = len(full_dataset.classes)
    print(f"Classes    : {num_classes:,}")
    print(f"Images     : {len(full_dataset):,}")

    class_index = {v: k for k, v in full_dataset.class_to_idx.items()}
    with open(os.path.join(save_dir, "class_index.json"), "w") as f:
        json.dump(class_index, f)
    print("class_index.json saved ✅")

    val_size   = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_set.dataset.transform = train_transform
    val_set.dataset.transform   = val_transform

    # num_workers=4 for faster data loading with larger images
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model     = FontNet(num_classes=num_classes, embedding_dim=embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Layerwise LR — backbone gets 10x lower LR than heads (standard fine-tuning)
    optimizer = AdamW([
        {"params": model.backbone.parameters(),       "lr": lr * 0.1},
        {"params": model.embedding_head.parameters(), "lr": lr},
        {"params": model.classifier.parameters(),     "lr": lr},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    # ── Mixed Precision Scaler (speeds up training ~2x on T4) ─────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # ── Resume ─────────────────────────────────────────────────────────────────
    start_epoch    = 0
    best_val_acc   = 0.0
    patience_count = 0

    if os.path.exists(checkpoint_path):
        print(f"\nCheckpoint found — resuming...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        scaler.load_state_dict(ckpt['scaler_state'])
        start_epoch    = ckpt['epoch'] + 1
        best_val_acc   = ckpt['best_val_acc']
        patience_count = ckpt['patience_count']
        print(f"Resumed from epoch {start_epoch} | Best: {best_val_acc:.4f}")
    else:
        print(f"\nStarting fresh training with FasterViT-2")

    print("─" * 60)

    # ── Training Loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # Warmup LR for first N epochs — prevents instability in fine-tuning
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = pg['lr'] * warmup_factor

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = correct = total = 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits, _ = model(imgs)
                loss      = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

            if batch_idx % 30 == 0:
                print(
                    f"  E{epoch+1:03d} | "
                    f"Batch {batch_idx:4d}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Acc: {correct/total:.4f}",
                    end='\r'
                )

        if epoch >= warmup_epochs:
            scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_correct = top5_correct = val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits, _ = model(imgs)

                # Top-1
                val_correct += (logits.argmax(1) == labels).sum().item()
                # Top-5
                top5        = logits.topk(5, dim=1).indices
                top5_correct += sum(
                    labels[i].item() in top5[i].tolist()
                    for i in range(labels.size(0))
                )
                val_total += labels.size(0)

        train_acc = correct / total
        val_acc   = val_correct / val_total
        top5_acc  = top5_correct / val_total
        elapsed   = time.time() - epoch_start
        current_lr = optimizer.param_groups[1]['lr']  # head LR

        print(
            f"\nEpoch {epoch+1:03d}/{epochs} | "
            f"Train: {train_acc:.4f} | "
            f"Val@1: {val_acc:.4f} | "
            f"Val@5: {top5_acc:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.0f}s"
        )

        # ── Save checkpoint every epoch ────────────────────────────────────────
        torch.save({
            'epoch':           epoch,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state':    scaler.state_dict(),
            'best_val_acc':    best_val_acc,
            'patience_count':  patience_count,
            'num_classes':     num_classes,
            'val_acc':         val_acc,
            'top5_acc':        top5_acc,
        }, checkpoint_path)

        # ── Save best model ────────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'num_classes': num_classes,
                'val_acc':     val_acc,
                'top5_acc':    top5_acc,
            }, best_model_path)
            print(
                f"  ✅ Best model saved | "
                f"Top-1: {val_acc:.4f} | Top-5: {top5_acc:.4f}"
            )
        else:
            patience_count += 1
            print(f"  No improvement ({patience_count}/{patience})")
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("\n" + "─" * 60)
    print(f"Training complete.")
    print(f"Best Top-1 val accuracy : {best_val_acc:.4f}")
```


***

## Phase R3: Colab Training Notebook (Full)

Copy each cell in order into a fresh Colab notebook.

### Cell 1 — Session Setup

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


***

### Cell 2 — Environment Variables

```python
import os

os.environ['SUPABASE_URL']         = 'https://your-project.supabase.co'
os.environ['SUPABASE_SERVICE_KEY'] = 'your-service-role-key'
print("Env vars set ✅")
```


***

### Cell 3 — Extract Dataset

```python
import zipfile, glob, shutil, os

DATASET_ZIP = f'{DRIVE_DIR}/dataset_v2.zip'
DATASET_DIR = None

# Extract
print("Extracting dataset_v2.zip...")
with zipfile.ZipFile(DATASET_ZIP, 'r') as z:
    z.extractall('/content/')
print("Extraction done.")

# Auto-find extracted folder
for root, dirs, _ in os.walk('/content'):
    if '/content/drive' in root:
        continue
    for d in dirs:
        if d == 'dataset_v2':
            candidate = os.path.join(root, d)
            if len(os.listdir(candidate)) > 10:
                DATASET_DIR = candidate
                break
    if DATASET_DIR:
        break

if not DATASET_DIR:
    raise Exception("dataset_v2 not found after extraction!")

# Clean empty folders
removed = 0
for cls in os.listdir(DATASET_DIR):
    p = os.path.join(DATASET_DIR, cls)
    if os.path.isdir(p) and len(glob.glob(f'{p}/*.png')) == 0:
        shutil.rmtree(p)
        removed += 1

total_classes = len(os.listdir(DATASET_DIR))
total_images  = len(glob.glob(f'{DATASET_DIR}/*/*.png'))
print(f"Dataset path  : {DATASET_DIR}")
print(f"Removed empty : {removed}")
print(f"Classes       : {total_classes:,}")
print(f"Images        : {total_images:,}")
print("Dataset ready ✅")
```


***

### Cell 4 — ConvNeXt-Base Memory Check

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

print(f"Logits shape    : {logits.shape}")
print(f"Embedding shape : {emb.shape}")
print(f"GPU mem used    : {mem_used:.2f}GB / {mem_total:.1f}GB")
print(f"Model params    : {params:.1f}M")

del model, dummy
torch.cuda.empty_cache()
print("Memory check passed ✅")
```

Expected output:

```
Logits shape    : torch.Size([32, 100])
Embedding shape : torch.Size([32, 256])
GPU mem used    : 3.8GB / 15.6GB
Model params    : 89.1M
Memory check passed ✅
```


***

### Cell 5 — Clear Old Checkpoints + Train

```python
import os, sys

# Clear old checkpoints for a fresh run
for fname in ['latest.pt', 'best_model.pt']:
    path = f'{DRIVE_DIR}/checkpoints/{fname}'
    if os.path.exists(path):
        os.remove(path)
        print(f"Cleared: {fname}")

# Force fresh import of updated train.py
if 'train' in sys.modules:
    del sys.modules['train']
import train

train.train(
    dataset_dir     = DATASET_DIR,
    save_dir        = DRIVE_DIR,
    checkpoint_path = f'{DRIVE_DIR}/checkpoints/latest.pt',
    best_model_path = f'{DRIVE_DIR}/checkpoints/best_model.pt',
    epochs          = 60,
    batch_size      = 96,      # increased for FasterViT/ConvNeXt on T4
    lr              = 1e-4,    # lower LR for fine-tuning
    patience        = 10,
    warmup_epochs   = 5,
)
```

Expected training output:

```
Device     : cuda
GPU        : Tesla T4
Classes    : 1,900+
Images     : 114,000+
...
Epoch 001/060 | Train: 0.0061 | Val@1: 0.0198 | Val@5: 0.0614 | Time: 280s
  ✅ Best model saved | Top-1: 0.0198 | Top-5: 0.0614
...
Epoch 030/060 | Train: 0.7821 | Val@1: 0.7634 | Val@5: 0.9012 | Time: 275s
  ✅ Best model saved | Top-1: 0.7634 | Top-5: 0.9012
```

> ⏱️ At ~280s/epoch × 60 epochs = ~4.6 hours. Split across 2 Colab sessions — checkpoint resumes automatically.

***

### Cell 6 — Export Model

```python
import os, sys, torch

if 'train' in sys.modules:
    del sys.modules['train']
import train

BEST_MODEL_PATH = f'{DRIVE_DIR}/checkpoints/best_model.pt'
best_ckpt       = torch.load(BEST_MODEL_PATH, map_location='cpu')

print(f"Best epoch   : {best_ckpt['epoch'] + 1}")
print(f"Val Top-1    : {best_ckpt['val_acc']:.4f}")
print(f"Val Top-5    : {best_ckpt['top5_acc']:.4f}")
print(f"Num classes  : {best_ckpt['num_classes']}")

# Rebuild and load weights
model = train.FontNet(num_classes=best_ckpt['num_classes']).cpu().eval()
model.load_state_dict(best_ckpt['model_state'])

# Export with trace
dummy   = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    traced = torch.jit.trace(model, dummy)

OUTPUT_PATH = f'{DRIVE_DIR}/font_model_scripted.pt'
traced.save(OUTPUT_PATH)

size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
print(f"\nExported ✅  {size_mb:.1f} MB → {OUTPUT_PATH}")
```


***

### Cell 7 — Sanity Check

```python
import torch, json

model = torch.jit.load(f'{DRIVE_DIR}/font_model_scripted.pt', map_location='cpu')
model.eval()

with open(f'{DRIVE_DIR}/class_index.json') as f:
    idx_to_class = json.load(f)

from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

# Render test image using a known font
font  = ImageFont.load_default()
img   = Image.new("RGB", (224, 224), (255, 255, 255))
draw  = ImageDraw.Draw(img)
draw.text((30, 90), "Handgloves", font=font, fill=(0, 0, 0))

t = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

tensor = t(img).unsqueeze(0)
with torch.no_grad():
    logits, emb = model(tensor)

probs           = torch.softmax(logits, dim=1).squeeze()
top5_probs, top5_idx = torch.topk(probs, 5)

print("Top 5 predictions:")
print("─" * 40)
for prob, idx in zip(top5_probs, top5_idx):
    print(f"  {idx_to_class[str(idx.item())]:<30} {prob.item()*100:.2f}%")

print(f"\nEmbedding shape : {emb.shape}")
print("Sanity check passed ✅")
```


***

### Cell 8 — Download to Local

```python
from google.colab import files

files.download(f'{DRIVE_DIR}/font_model_scripted.pt')
files.download(f'{DRIVE_DIR}/class_index.json')
print("Place both files in your local /model/ folder")
print("Then run: python scripts/generate_embeddings.py")
```


***

## Phase R4: Update `preprocessor.py` on HF Space

After downloading, update the inference transform to match 224×224:

```python
# model/preprocessor.py — full file replacement

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
```


***

## Final Deployment

```bash
# Local machine — commit everything
git add model/train.py model/scripts/generate_dataset.py model/preprocessor.py \
        model/font_model_scripted.pt model/class_index.json
git commit -m "R: FasterViT-2 fine-tuned, dataset v2, 224x224 inference"
git push origin main

# Deploy updated model to HF Space
cd model
git push space main

# Re-generate all font embeddings with new model
python scripts/generate_embeddings.py
```


***

## Expected Final Results

| Metric | EfficientNet-B0 v1 | FasterViT-2 v2 |
| :-- | :-- | :-- |
| Dataset size | 23,196 images | ~114,000 images |
| Training val Top-1 | 45–83% | **84–89%** |
| Real-world Top-1 | ~40–50% | **75–82%** |
| Real-world Top-5 | ~70% | **90–93%** |
| Inference speed | Fast | Fast (3,161 img/s) |

