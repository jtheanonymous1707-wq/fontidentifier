<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# ---------------------------------------------------------------------------

ModuleNotFoundError                       Traceback (most recent call last)

[/tmp/ipykernel_619/783220733.py](https://localhost:8080/#) in <cell line: 0>()
----> 1 import train
2 train.train(
3     dataset_dir     = '/content/data/dataset', \# Path in Colab
4     save_dir        = DRIVE_DIR,
5     checkpoint_path = f'{DRIVE_DIR}/checkpoints/latest.pt',

ModuleNotFoundError: No module named 'train'

---------------------------------------------------------------------------
NOTE: If your import is failing due to a missing package, you can
manually install dependencies using either !pip or !apt.

To view examples of installing some common dependencies, click the
"Open Examples" button below.
---------------------------------------------------------------------------

The `sys.path.insert` from the previous session didn't carry over — Colab loses all variable state when it reconnects. You need to re-run the path setup. Here's the full **session restart cell** to always run first:

***

## Cell 1 — Run This Every Time Colab Reconnects

```python
# ── 1. Mount Drive ────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ── 2. Pull latest code from GitHub ──────────────────────
import os

REPO_DIR  = '/content/font-identifier'
DRIVE_DIR = '/content/drive/MyDrive/font-identifier'

if os.path.exists(REPO_DIR):
    !git -C {REPO_DIR} pull   # update if already cloned
else:
    !git clone https://github.com/YOUR_USERNAME/font-identifier.git {REPO_DIR}

# ── 3. Add model folder to Python path ───────────────────
import sys
MODEL_DIR = f'{REPO_DIR}/model'
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

# ── 4. Install dependencies ───────────────────────────────
!pip install -q timm supabase python-dotenv

# ── 5. Verify GPU ─────────────────────────────────────────
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"sys.path includes model: {MODEL_DIR in sys.path}")

# ── 6. Test import ────────────────────────────────────────
import train
print(f"train.py loaded ✅")
print(f"FontNet available: {hasattr(train, 'FontNet')}")
```


***

## Cell 2 — Then Run Training

```python
import os

os.environ['GOOGLE_FONTS_API_KEY'] = 'your-key'
os.environ['SUPABASE_URL']         = 'https://your-project.supabase.co'
os.environ['SUPABASE_SERVICE_KEY'] = 'your-service-role-key'

train.train(
    dataset_dir     = '/content/data/dataset',
    save_dir        = DRIVE_DIR,
    checkpoint_path = f'{DRIVE_DIR}/checkpoints/latest.pt',
    best_model_path = f'{DRIVE_DIR}/checkpoints/best_model.pt',
    epochs          = 80,
    batch_size      = 128,
    lr              = 3e-4,
    patience        = 12,
)
```


***

## Why This Keeps Happening

Colab resets the **entire Python runtime** on every reconnect — all `sys.path` changes, imports, and variables are wiped. The fix is to consolidate everything into **one "session setup" cell** that you always run first before anything else.

A good habit: keep this setup cell at the very top of your notebook with a comment:

```python
# ⚠️ RUN THIS FIRST after every reconnect
```

