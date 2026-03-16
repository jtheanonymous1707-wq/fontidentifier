<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# why are we downloading the font in colab? can't we do it locally? The font has to be able to download from the website. i think we do google fonts first.

Great catch — yes, you can and **should** do the font downloading and dataset generation locally. Colab should only be used for the actual GPU training. Here's the corrected approach:

***

## Why Local Is Better for This

| Task | Where to Run | Reason |
| :-- | :-- | :-- |
| Download fonts | ✅ Local | Colab session can die mid-download, wasting time |
| Generate dataset images | ✅ Local | Pure CPU work, no GPU needed |
| Model training | ✅ Colab | Needs T4 GPU |
| Export model | ✅ Colab | Already there after training |


***

## Revised Plan: Do This Locally First

### Step 1 — Run Font Download Locally

Inside `/model`, run:

```bash
cd model
source venv/bin/activate
python scripts/download_fonts.py
```

This saves all `.ttf` files into `model/data/fonts/`. Since you're starting with **Google Fonts only**, your `download_fonts.py` from Phase 3 already handles this — just make sure it points to your local `data/fonts/` folder:

```python
FONTS_DIR = "data/fonts"   # relative to /model — saves locally
```

Expected result:

```
Fetching Google Fonts API...
Downloading 1521 fonts...
Progress: 100/1521
Progress: 200/1521
...
Done. 1521 fonts saved to data/fonts/
```


***

### Step 2 — Generate Dataset Locally

```bash
python scripts/generate_dataset.py
```

This renders ~18,000 images into `model/data/dataset/`. Takes **15–20 minutes** on a normal laptop CPU — go grab a coffee.

```
Generating dataset for 1521 fonts...
Progress: 100/1521 fonts
Progress: 200/1521 fonts
...
Dataset ready: 18,252 images across 1521 font classes
```


***

### Step 3 — Zip and Upload to Google Drive

The dataset folder is too large to `git push`, so zip it and upload manually to Drive:

```bash
cd model
zip -r dataset.zip data/dataset/
zip -r fonts.zip data/fonts/
```

Then drag both zips into your Google Drive folder `font-identifier/` via [drive.google.com](https://drive.google.com).

***

### Step 4 — Unzip in Colab at Training Time

At the start of your Colab session, instead of re-downloading everything, just unzip from Drive:

```python
# Replace Steps 6 & 7 from the previous guide with this:
import zipfile

print("Extracting dataset from Drive...")
with zipfile.ZipFile(f'{DRIVE_DIR}/dataset.zip', 'r') as z:
    z.extractall('/content/')

with zipfile.ZipFile(f'{DRIVE_DIR}/fonts.zip', 'r') as z:
    z.extractall('/content/')

DATASET_DIR = '/content/data/dataset'
FONTS_DIR   = '/content/data/fonts'

import glob
print(f"Fonts: {len(glob.glob(FONTS_DIR + '/*.ttf'))}")
print(f"Images: {len(glob.glob(DATASET_DIR + '/*/*.png'))}")
```

Extraction takes ~2–3 minutes, which is far faster than re-downloading and re-rendering every session.

***

## Regarding Your Point About Font Downloads on the Website

You're thinking ahead correctly — for the live web app, users identify fonts **from images they upload**, not from downloading font files. The font `.ttf` files are only needed for:

1. **Training** — to render synthetic text images (local + Colab only)
2. **Embedding generation** — to create the 256-dim fingerprint per font (local only)

Once both are done, the `.ttf` files are **never needed again** in production. The website just stores font metadata (name, category, Google Fonts URL) and embeddings in Supabase — no font files hosted anywhere.

***

## Updated Execution Order

```
Local machine:
  1. python scripts/download_fonts.py     ← download TTFs
  2. python scripts/generate_dataset.py   ← render images
  3. zip dataset + fonts → upload to Drive

Google Colab:
  4. Unzip from Drive
  5. Train model (Steps 8–13 from previous guide)
  6. Export font_model_scripted.pt

Local machine:
  7. Download model from Drive → git push
```

This way Colab only ever does what it's good at — GPU training. Everything else stays on your machine where it's stable and resumable.

