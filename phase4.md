<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# continue phase 4

Here's the full Phase 4 implementation guide.

***

## Phase 4: FastAPI Inference Service on Hugging Face Spaces

### Step 1 — Set Up the HF Space Git Remote

Hugging Face Spaces uses Git for deployment — every push triggers a rebuild. Link your `/model` folder to the HF Space repo:[^1]

```bash
cd model

# Initialize git if not already
git init

# Add HF Space as a remote
git remote add space https://huggingface.co/spaces/YOUR_HF_USERNAME/font-identifier

# Also keep GitHub as origin
git remote add origin https://github.com/YOUR_USERNAME/font-identifier.git
```

Your `/model` folder will push to **two remotes**: GitHub for version control, HF Spaces for live deployment.

***

### Step 2 — Final Folder Structure

Before writing any code, make sure `/model` looks exactly like this:

```
model/
├── app.py                        ← FastAPI entrypoint
├── predictor.py                  ← Model loading + inference logic
├── preprocessor.py               ← Image preprocessing pipeline
├── font_model_scripted.pt        ← Trained model (from Phase 3)
├── class_index.json              ← {0: "Roboto", 1: "Open Sans", ...}
├── requirements.txt
├── Dockerfile
├── .env                          ← Local only, never pushed
├── .gitignore
└── scripts/                      ← Seeding scripts from Phase 2 & 3
```


***

### Step 3 — Write the Preprocessor

Create `/model/preprocessor.py` — isolates all image prep logic so it's reusable and testable:

```python
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
        background.paste(img, mask=img.split()[^3])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Auto-level: enhance contrast on low-contrast images
    img = ImageOps.autocontrast(img, cutoff=2)

    # Mild denoise for photos
    img = img.filter(ImageFilter.MedianFilter(size=1))

    return INFERENCE_TRANSFORM(img).unsqueeze(0)  # shape: [1, 3, 64, 128]
```


***

### Step 4 — Write the Predictor

Create `/model/predictor.py` — handles model loading once at startup and exposes a clean `predict()` function:

```python
import os, json, torch
import numpy as np

MODEL_PATH      = "font_model_scripted.pt"
CLASS_INDEX_PATH = "class_index.json"

# Load once at module import — not on every request
print("Loading font model...")
_model = torch.jit.load(MODEL_PATH, map_location="cpu")
_model.eval()

with open(CLASS_INDEX_PATH) as f:
    _idx_to_class: dict = json.load(f)   # {"0": "Roboto", "1": "Open Sans", ...}

print(f"Model loaded. {len(_idx_to_class)} font classes.")

def predict(tensor: "torch.Tensor") -> dict:
    """
    Returns top-5 predicted fonts with confidence scores + the 256-dim embedding.
    """
    with torch.no_grad():
        logits, embedding = _model(tensor)

    # Top-5 predictions
    probs      = torch.softmax(logits, dim=1).squeeze()
    top5_probs, top5_idx = torch.topk(probs, k=5)

    top5 = [
        {
            "font_name": _idx_to_class[str(idx.item())],
            "confidence": round(prob.item() * 100, 2)
        }
        for prob, idx in zip(top5_probs, top5_idx)
    ]

    return {
        "top5": top5,
        "embedding": embedding.squeeze().tolist()   # 256-dim list
    }
```


***

### Step 5 — Write the FastAPI App

Create `/model/app.py` — the main entrypoint with all endpoints:

```python
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from supabase import create_client
from preprocessor import preprocess_image
from predictor import predict
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Font Identifier API",
    description="Upload a cropped text image, get back font matches.",
    version="1.0.0"
)

# Allow requests from your Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-app.vercel.app"    # update this after Vercel deploy
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Supabase client for vector similarity search
supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"]
)

# ── Endpoints ────────────────────────────────────────────────

@app.get("/ping")
def ping():
    """Warmup endpoint — called on page load to wake the Space."""
    return {"status": "awake", "model": "font-identifier-v1"}


@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    """
    Main endpoint. Accepts a cropped image, returns top 5 font matches
    with similarity scores pulled from Supabase pgvector.
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload JPEG, PNG, or WebP."
        )

    # Read and validate file size (max 5MB)
    image_bytes = await file.read()
    if len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 5MB.")

    try:
        # Preprocess → infer
        tensor     = preprocess_image(image_bytes)
        prediction = predict(tensor)
        embedding  = prediction["embedding"]
        top5_local = prediction["top5"]

        # Vector similarity search in Supabase
        result = supabase.rpc(
            "match_fonts",
            {
                "query_embedding": embedding,
                "match_count": 5
            }
        ).execute()

        matches = result.data if result.data else []

        # Merge local top5 confidence with Supabase metadata
        # Supabase result is the authoritative source for font metadata
        response = {
            "matches": [
                {
                    "id": match["id"],
                    "name": match["name"],
                    "category": match["category"],
                    "google_fonts_url": match["google_fonts_url"],
                    "preview_url": match["preview_url"],
                    "similarity": round(match["similarity"] * 100, 1),
                    "confidence": next(
                        (f["confidence"] for f in top5_local
                         if f["font_name"] == match["name"]), None
                    )
                }
                for match in matches
            ]
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/feedback")
async def feedback(payload: dict):
    """
    Stores user correction (thumbs up/down) into Supabase feedback table.
    Used to build the retraining dataset over time.
    """
    required = {"image_url", "predicted_font_id", "is_correct"}
    if not required.issubset(payload):
        raise HTTPException(status_code=422, detail="Missing required fields.")

    supabase.table("feedback").insert({
        "image_url":          payload["image_url"],
        "predicted_font_id":  payload["predicted_font_id"],
        "correct_font_id":    payload.get("correct_font_id"),
        "is_correct":         payload["is_correct"]
    }).execute()

    return {"status": "feedback recorded"}
```


***

### Step 6 — Write the Dockerfile

Create `/model/Dockerfile` — HF Spaces requires the app to run on **port 7860**:[^2]

```dockerfile
FROM python:3.11-slim

# System dependencies for Pillow + PyTorch
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install CPU-only PyTorch (much smaller than CUDA build ~200MB vs ~2GB)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app.py predictor.py preprocessor.py ./
COPY font_model_scripted.pt class_index.json ./

# HF Spaces runs on port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
```

Update `/model/requirements.txt` to exclude torch (installed separately in Dockerfile):

```
fastapi
uvicorn[standard]
pillow
timm
supabase
python-dotenv
python-multipart
numpy
```


***

### Step 7 — Configure HF Space Secrets

Never push `.env` to HF Spaces. Add secrets through the UI instead:

1. Go to your HF Space → **Settings → Variables and Secrets**
2. Click **New Secret** and add:
    - `SUPABASE_URL` → your Supabase project URL
    - `SUPABASE_SERVICE_KEY` → your service role key
3. These are injected as environment variables at runtime — `os.environ["SUPABASE_URL"]` will work as expected

***

### Step 8 — Set Up `.gitignore`

Create `/model/.gitignore`:

```
.env
venv/
__pycache__/
*.pyc
data/          # dataset is too large to push
```

> ⚠️ `font_model_scripted.pt` **should** be committed — it needs to be in the container. At ~20MB it's within Git's limits.

***

### Step 9 — Test Locally First

Before pushing to HF, test locally with Docker:

```bash
cd model

# Build the image
docker build -t font-identifier .

# Run with your local .env injected
docker run --env-file .env -p 7860:7860 font-identifier
```

Test each endpoint:

```bash
# 1. Ping (warmup)
curl http://localhost:7860/ping

# 2. Identify a font (use any cropped text image)
curl -X POST http://localhost:7860/identify \
  -F "file=@test_image.png"

# 3. Feedback
curl -X POST http://localhost:7860/feedback \
  -H "Content-Type: application/json" \
  -d '{"image_url":"test.png","predicted_font_id":1,"is_correct":true}'
```

Expected `/identify` response:

```json
{
  "matches": [
    {
      "id": 42,
      "name": "Roboto",
      "category": "sans-serif",
      "google_fonts_url": "https://fonts.google.com/specimen/Roboto",
      "preview_url": null,
      "similarity": 94.2,
      "confidence": 87.5
    },
    ...
  ]
}
```


***

### Step 10 — Deploy to Hugging Face Spaces

Once local tests pass:

```bash
cd model

git add app.py predictor.py preprocessor.py Dockerfile requirements.txt \
        font_model_scripted.pt class_index.json .gitignore

git commit -m "Phase 4: FastAPI inference service with CORS, warmup, feedback endpoint"

# Push to HF Spaces (triggers auto-build)
git push space main

# Also push to GitHub
git push origin main
```

Monitor the build in the **Logs** tab of your HF Space. Build takes ~3–5 minutes. Watch for:

```
✓ Installing packages...
✓ Loading font model...
✓ Model loaded. 1521 font classes.
✓ Uvicorn running on http://0.0.0.0:7860
```

Your live API URL will be:

```
https://YOUR_HF_USERNAME-font-identifier.hf.space
```


***

### Step 11 — Note the Space Sleep Behavior

HF free tier Spaces sleep after **~15 minutes** of inactivity. Your Next.js frontend handles this gracefully with the warmup call (Phase 5), but test the cold start:[^1]

```bash
# After the Space has been idle, time the first request
time curl https://YOUR_HF_USERNAME-font-identifier.hf.space/ping
# Cold start: ~8–15 seconds
# Warm: <500ms
```

Account for this in the frontend by showing a **"Warming up model..."** status when the ping takes longer than 2 seconds.

***

## Phase 4 Checklist

- [ ] `/model` folder linked to both GitHub and HF Spaces remotes
- [ ] `preprocessor.py` written with RGBA/grayscale handling and auto-contrast
- [ ] `predictor.py` loads model once at startup, returns top-5 + embedding
- [ ] `app.py` has `/ping`, `/identify`, `/feedback` endpoints with validation
- [ ] CORS configured for `localhost:3000` and your Vercel URL
- [ ] Dockerfile uses CPU-only PyTorch and runs on port 7860
- [ ] HF Space Secrets configured (no hardcoded keys)
- [ ] `.gitignore` excludes `.env` and `data/`
- [ ] Local Docker test passes all three endpoints
- [ ] Successfully deployed to HF Spaces — logs show model loaded
- [ ] Cold start time measured and noted for frontend handling
- [ ] Both `git push space main` and `git push origin main` done

You're now ready for **Phase 5: Frontend (Next.js on Vercel)** — where you'll build the crop UI, wire up all these endpoints, and display the results as polished font cards.
<span style="display:none">[^10][^11][^12][^13][^14][^15][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://www.youtube.com/watch?v=0v9ZsleUuEg

[^2]: https://huggingface.co/blog/HemanthSai7/deploy-applications-on-huggingface-spaces

[^3]: https://discuss.huggingface.co/t/how-backgroundtasks-fastapi-works-in-huggingface-space/109597

[^4]: https://www.runpod.io/articles/guides/deploy-hugging-face-docker

[^5]: https://discuss.huggingface.co/t/streamlit-fastapi-deployment-issue/86217

[^6]: https://pyimagesearch.com/2025/11/17/fastapi-docker-deployment-preparing-onnx-ai-models-for-aws-lambda/

[^7]: https://blog.devgenius.io/build-machine-learning-apps-with-hugging-faces-docker-spaces-b36fbe737631

[^8]: https://revs.runtime-revolution.com/running-deep-learning-models-as-applications-with-fastapi-ecac57239e64

[^9]: https://www.youtube.com/watch?v=Sx_MwcBQGOg

[^10]: https://dev.to/zestminds_technologies_c1/fastapi-setup-guide-for-2025-requirements-structure-deployment-1gd

[^11]: https://discuss.huggingface.co/t/fastapi-routing-issue-in-hugging-face-spaces/54962

[^12]: https://discuss.huggingface.co/t/how-to-obtain-faster-inference-endpoint/137032

[^13]: https://discuss.huggingface.co/t/deploying-nginx-in-docker-with-streamlit-and-fastapi/95068

[^14]: https://python.plainenglish.io/fastapi-build-scalable-python-apis-for-ai-and-ml-applications-6a0365ab572e

[^15]: https://huggingface.co/docs/hub/en/spaces-sdks-docker-examples

