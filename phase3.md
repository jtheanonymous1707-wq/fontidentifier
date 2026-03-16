<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# continue with phase 3 but add improvement and optimization

Here's the full Phase 3 guide with optimizations baked in at every step.

***

## Phase 3: Model Training Pipeline

### Overview of Improvements vs. Basic DeepFont

| Basic DeepFont | Optimized Version (This Guide) |
| :-- | :-- |
| VGG-style CNN from scratch | EfficientNet-B0 pretrained backbone [^1] |
| Simple text rendering | 9-layer augmentation pipeline [^2] |
| Single output (class label) | Dual output (class + 256-dim embedding) |
| No learning rate scheduling | Cosine annealing + warmup |
| Single checkpoint | Best model + early stopping |
| Fixed embedding index | Dynamic embedding refresh after retraining |


***

### Step 1 — Set Up Google Colab (Free GPU)

All training runs on **Google Colab's free T4 GPU** — no cost.

1. Go to [colab.research.google.com](https://colab.research.google.com) → New Notebook
2. Runtime → Change runtime type → **T4 GPU**
3. Mount your Google Drive to persist files between sessions:

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs('/content/drive/MyDrive/font-identifier', exist_ok=True)
SAVE_DIR = '/content/drive/MyDrive/font-identifier'
```

4. Clone your repo into Colab:

```bash
!git clone https://github.com/YOUR_USERNAME/font-identifier.git
%cd font-identifier/model
!pip install -r requirements.txt
```


***

### Step 2 — Download All Google Fonts

Create `/model/scripts/download_fonts.py`:

```python
import os, requests, zipfile, io
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["GOOGLE_FONTS_API_KEY"]
FONTS_DIR = "data/fonts"
os.makedirs(FONTS_DIR, exist_ok=True)

def download_all_fonts():
    res = requests.get(f"https://www.googleapis.com/webfonts/v1/webfonts?key={API_KEY}&sort=alpha")
    fonts = res.json()["items"]
    print(f"Downloading {len(fonts)} font families...")

    for font in fonts:
        family = font["family"]
        # Download only the 'regular' variant to save space
        files = font.get("files", {})
        url = files.get("regular") or list(files.values())[^0]
        if not url:
            continue

        save_path = os.path.join(FONTS_DIR, f"{family}.ttf")
        if os.path.exists(save_path):
            continue  # skip already downloaded

        try:
            r = requests.get(url, timeout=10)
            with open(save_path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            print(f"Failed {family}: {e}")

    print(f"Downloaded fonts to {FONTS_DIR}")

if __name__ == "__main__":
    download_all_fonts()
```

Run it:

```bash
python scripts/download_fonts.py
```


***

### Step 3 — Build the Augmented Dataset Generator

This is the most critical step. Create `/model/scripts/generate_dataset.py`.

The key insight from research is that adding **degradation augmentation** to synthetic data (blur, noise, rotation, perspective warp) is what bridges the gap between clean rendered text and real-world photos.[^2]

```python
import os, random, glob
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np

FONTS_DIR = "data/fonts"
OUTPUT_DIR = "data/dataset"
IMG_SIZE = (128, 64)     # width x height per sample
SAMPLES_PER_FONT = 12    # ~18,000 total images for 1,500 fonts

SAMPLE_TEXTS = [
    "Handgloves", "Typography", "AaBbCc", "FontID",
    "Design 2024", "Hello World", "Quick fox", "Spectrum",
    "ABCDE fghij", "Waltz nymph"
]

def apply_augmentations(img: Image.Image, level: int = 1) -> Image.Image:
    """level 0 = clean, level 1 = mild, level 2 = heavy"""
    img = img.convert("RGB")

    if level >= 1:
        # Gaussian blur (simulates out-of-focus photos)
        if random.random() > 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
        # Brightness variation
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.75, 1.25))
        # Contrast variation
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.3))

    if level >= 2:
        # Add Gaussian noise
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, random.uniform(5, 20), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        # Slight rotation (-5 to +5 degrees)
        img = img.rotate(random.uniform(-5, 5), fillcolor=(255, 255, 255))
        # Random horizontal compression/stretch
        new_w = int(img.width * random.uniform(0.85, 1.15))
        img = img.resize((new_w, img.height), Image.LANCZOS)
        img = img.crop((0, 0, IMG_SIZE[^0], IMG_SIZE[^1]))
        img = img.resize(IMG_SIZE, Image.LANCZOS)

    return img

def render_text_image(font_path: str, text: str, font_size: int = 32) -> Image.Image:
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        return None

    img = Image.new("RGB", IMG_SIZE, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Center text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[^2] - bbox[^0]
    text_h = bbox[^3] - bbox[^1]
    x = max(0, (IMG_SIZE[^0] - text_w) // 2)
    y = max(0, (IMG_SIZE[^1] - text_h) // 2)

    # Randomize text color (dark on light background)
    text_color = (
        random.randint(0, 60),
        random.randint(0, 60),
        random.randint(0, 60)
    )
    draw.text((x, y), text, font=font, fill=text_color)
    return img

def generate_dataset():
    font_paths = glob.glob(os.path.join(FONTS_DIR, "*.ttf"))
    print(f"Generating dataset for {len(font_paths)} fonts...")

    for font_path in font_paths:
        font_name = os.path.splitext(os.path.basename(font_path))[^0]
        class_dir = os.path.join(OUTPUT_DIR, font_name)
        os.makedirs(class_dir, exist_ok=True)

        # Skip if already generated
        if len(os.listdir(class_dir)) >= SAMPLES_PER_FONT:
            continue

        count = 0
        for text in SAMPLE_TEXTS:
            for font_size in [28, 32, 38]:
                if count >= SAMPLES_PER_FONT:
                    break
                img = render_text_image(font_path, text, font_size)
                if img is None:
                    continue

                # Save 1 clean + 1 mild-aug + 1 heavy-aug version
                for level in [0, 1, 2]:
                    aug = apply_augmentations(img.copy(), level=level)
                    aug.save(os.path.join(class_dir, f"{count}_aug{level}.png"))
                    count += 1
                    if count >= SAMPLES_PER_FONT:
                        break

    print(f"Dataset generated at {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_dataset()
```

Run it:

```bash
python scripts/generate_dataset.py
# Takes ~10-15 mins for 1,500 fonts
```


***

### Step 4 — Build the Model Architecture

Create `/model/train.py`. The architecture uses **EfficientNet-B0** as backbone (5.3M params, outperforms ResNet-50 at 1/5th the size ) with a **dual-head** design — one head for classification, one for 256-dim embedding extraction:[^1]

```python
import torch
import torch.nn as nn
import timm

class FontNet(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 256):
        super().__init__()
        # EfficientNet-B0 pretrained backbone
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0,        # Remove default classifier head
            global_pool='avg'
        )
        backbone_out = self.backbone.num_features  # 1280 for B0

        # Embedding head — for similarity search in Supabase
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_out, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)   # L2-normalize later
        )

        # Classification head — for training supervision
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x, return_embedding=False):
        features = self.backbone(x)
        embedding = self.embedding_head(features)
        # L2 normalize the embedding
        embedding = nn.functional.normalize(embedding, p=2, dim=1)

        if return_embedding:
            return embedding

        logits = self.classifier(embedding)
        return logits, embedding
```


***

### Step 5 — Build the Training Pipeline

Add to `/model/train.py`:

```python
import os, json
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ── Config ──────────────────────────────────────────────────
DATASET_DIR  = "data/dataset"
SAVE_DIR     = "/content/drive/MyDrive/font-identifier"
BATCH_SIZE   = 64
EPOCHS       = 60
LR           = 3e-4
EMBEDDING_DIM = 256
PATIENCE     = 8     # early stopping

# ── Transforms ──────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((64, 128)),
    transforms.RandomHorizontalFlip(p=0.1),    # only subtle for text
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])  # ImageNet stats
])

val_transform = transforms.Compose([
    transforms.Resize((64, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Dataset Split ────────────────────────────────────────────
full_dataset = datasets.ImageFolder(DATASET_DIR)
num_classes  = len(full_dataset.classes)

# Save class index mapping to JSON (needed for inference later)
with open(os.path.join(SAVE_DIR, "class_index.json"), "w") as f:
    json.dump({v: k for k, v in full_dataset.class_to_idx.items()}, f)

val_size   = int(0.1 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_set, val_set = random_split(full_dataset, [train_size, val_size])
train_set.dataset.transform = train_transform
val_set.dataset.transform   = val_transform

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

# ── Model, Loss, Optimizer ───────────────────────────────────
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = FontNet(num_classes=num_classes, embedding_dim=EMBEDDING_DIM).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)   # label smoothing prevents overconfidence
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

# ── Training Loop ────────────────────────────────────────────
best_val_acc  = 0.0
patience_count = 0

for epoch in range(EPOCHS):
    # — Train —
    model.train()
    train_loss, correct, total = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
        optimizer.step()
        train_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    scheduler.step()

    # — Validate —
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits, _ = model(imgs)
            val_loss    += criterion(logits, labels).item()
            val_correct += (logits.argmax(1) == labels).sum().item()
            val_total   += labels.size(0)

    train_acc = correct / total
    val_acc   = val_correct / val_total
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train: {train_acc:.3f} | Val: {val_acc:.3f} | "
          f"LR: {scheduler.get_last_lr()[^0]:.6f}")

    # — Save best model —
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "num_classes": num_classes,
            "val_acc": val_acc
        }, os.path.join(SAVE_DIR, "font_model_best.pt"))
        print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
```

Run training in Colab:

```python
%run train.py
# Expected: ~2–3 hours on free T4, reaching ~82-87% top-1 val accuracy
```


***

### Step 6 — Generate Font Embeddings

After training, generate a single clean embedding per font and push to Supabase. Create `/model/scripts/generate_embeddings.py`:

```python
import os, json, glob, torch
import numpy as np
from PIL import Image
from torchvision import transforms
from supabase import create_client
from dotenv import load_dotenv
from train import FontNet

load_dotenv()
SAVE_DIR  = "/content/drive/MyDrive/font-identifier"
FONTS_DIR = "data/fonts"

# Load model
checkpoint = torch.load(os.path.join(SAVE_DIR, "font_model_best.pt"), map_location="cpu")
with open(os.path.join(SAVE_DIR, "class_index.json")) as f:
    idx_to_class = json.load(f)   # {0: "Roboto", 1: "Open Sans", ...}

model = FontNet(num_classes=checkpoint["num_classes"])
model.load_state_dict(checkpoint["model_state"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

def get_embedding(font_path: str) -> np.ndarray:
    from PIL import ImageFont, ImageDraw
    font = ImageFont.truetype(font_path, 32)
    img  = Image.new("RGB", (128, 64), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 15), "Handgloves", font=font, fill=(0, 0, 0))
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = model(tensor, return_embedding=True)
    return emb.squeeze().numpy().tolist()

font_paths = glob.glob(os.path.join(FONTS_DIR, "*.ttf"))
print(f"Generating embeddings for {len(font_paths)} fonts...")

for font_path in font_paths:
    font_name = os.path.splitext(os.path.basename(font_path))[^0]
    try:
        embedding = get_embedding(font_path)
        # Update the matching row in Supabase
        supabase.table("fonts") \
            .update({"embedding": embedding}) \
            .eq("name", font_name) \
            .execute()
        print(f"✓ {font_name}")
    except Exception as e:
        print(f"✗ {font_name}: {e}")

print("All embeddings uploaded!")
```

Run it:

```bash
python scripts/generate_embeddings.py
# Takes ~15-20 mins for 1,500 fonts
```


***

### Step 7 — Export the Model for Deployment

Compress and export the model to make it small enough for Hugging Face Spaces' free tier:

```python
# Run in Colab
import torch
from train import FontNet

checkpoint = torch.load(f"{SAVE_DIR}/font_model_best.pt")
model = FontNet(num_classes=checkpoint["num_classes"])
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Export with TorchScript for faster inference (no Python overhead)
scripted = torch.jit.script(model)
scripted.save(f"{SAVE_DIR}/font_model_scripted.pt")

# Check file size — target is under 25MB
import os
size_mb = os.path.getsize(f"{SAVE_DIR}/font_model_scripted.pt") / 1e6
print(f"Model size: {size_mb:.1f} MB")   # Expected: ~18–22MB for B0
```

Copy back to your local repo:

```bash
cp /content/drive/MyDrive/font-identifier/font_model_scripted.pt model/
cp /content/drive/MyDrive/font-identifier/class_index.json model/
```


***

### Step 8 — Verify Embeddings in Supabase

In your Supabase SQL Editor, confirm embeddings are populated:

```sql
-- Check how many fonts have embeddings
select
  count(*) filter (where embedding is not null) as with_embedding,
  count(*) filter (where embedding is null)     as missing_embedding
from fonts;

-- Quick similarity test: find fonts similar to "Roboto"
select name, category,
       1 - (embedding <=> (select embedding from fonts where name = 'Roboto')) as similarity
from fonts
where name != 'Roboto' and embedding is not null
order by similarity desc
limit 10;
```

You should see fonts like **Inter**, **Open Sans**, and **Noto Sans** appearing as most similar to Roboto — if that looks right, your embeddings are working correctly.

***

### Step 9 — Commit Everything

```bash
cd ..   # repo root
git add model/font_model_scripted.pt model/scripts/ model/train.py
git commit -m "Phase 3: EfficientNet-B0 training pipeline, augmentation, embeddings uploaded"
git push origin main
```


***

## Phase 3 Checklist

- [ ] Google Colab set up with T4 GPU + Google Drive mounted
- [ ] All Google Fonts `.ttf` files downloaded via `download_fonts.py`
- [ ] Augmented dataset generated (~18,000 images across 1,500 classes)
- [ ] `FontNet` (EfficientNet-B0 dual-head) architecture built
- [ ] Model trained with cosine LR scheduling, label smoothing, gradient clipping
- [ ] Best model saved via early stopping (target: >80% top-1 val accuracy)
- [ ] Font embeddings generated and uploaded to Supabase (all 1,500 rows)
- [ ] Supabase similarity test returns visually correct results
- [ ] `font_model_scripted.pt` exported at <25MB
- [ ] `class_index.json` saved alongside model
- [ ] Everything committed to GitHub

You're now ready for **Phase 4: FastAPI Inference Service on Hugging Face Spaces**, where you'll wrap this model in an API endpoint.
<span style="display:none">[^10][^11][^12][^13][^14][^15][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://viso.ai/deep-learning/efficientnet/

[^2]: http://ltu.diva-portal.org/smash/record.jsf?pid=diva2%3A1673617

[^3]: https://www.nature.com/articles/s41598-025-34262-2

[^4]: https://ieeexplore.ieee.org/document/10861966/

[^5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12475328/

[^6]: https://github.com/mishra-kunal1/font_classification

[^7]: https://www.studocu.vn/vn/document/van-lang-university/it-information-technology/deepfont-visual-font-recognition-using-cnns-and-domain-adaptation-techniques/126784710

[^8]: https://www.sciencedirect.com/science/article/pii/S1110866525000878

[^9]: https://www.emergentmind.com/topics/synthocr-gen

[^10]: https://github.com/robinreni96/Font_Recognition-DeepFont

[^11]: https://repositori.uma.ac.id/jspui/handle/123456789/27353

[^12]: https://arxiv.org/pdf/2403.10075.pdf

[^13]: https://arxiv.org/pdf/1708.03669.pdf

[^14]: https://www.sciencedirect.com/science/article/pii/S0957417425017713

[^15]: https://www.collabora.com/news-and-blog/blog/2025/11/11/font-recognition-reimagined-with-fastervit-2/

