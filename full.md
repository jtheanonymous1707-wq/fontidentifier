<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Let's get started. Write me a full implementation plan to guide me step by step

Here's your full implementation plan broken into **6 phases**, ordered from setup to deployment.

***

## Phase 1: Project Setup \& Repo Structure

**Goal:** Scaffold the monorepo and connect all services.

1. Create a GitHub repo with two folders: `/web` (Next.js) and `/model` (Python)
2. Init Next.js app in `/web` with TypeScript: `npx create-next-app@latest web --typescript`
3. Install dependencies: `react-image-crop`, `@supabase/supabase-js`, `axios`
4. Create a **Supabase project** at supabase.com — note your `SUPABASE_URL` and `ANON_KEY`
5. Create a **Hugging Face account** and create a new Space (type: Docker or Gradio)
6. Set up `.env.local` in `/web`:

```
NEXT_PUBLIC_SUPABASE_URL=...
NEXT_PUBLIC_SUPABASE_ANON_KEY=...
NEXT_PUBLIC_HF_API_URL=https://your-space.hf.space
```


***

## Phase 2: Font Database Setup (Supabase)

**Goal:** Store all font metadata and embeddings for similarity search.

1. **Enable pgvector** extension in Supabase SQL editor:

```sql
create extension vector;
```

2. Create the `fonts` table:

```sql
create table fonts (
  id serial primary key,
  name text not null,
  family text,
  category text,          -- serif, sans-serif, monospace, display
  google_fonts_url text,
  preview_url text,
  embedding vector(256)   -- DeepFont feature vector dimension
);
```

3. Create an index for fast similarity search:

```sql
create index on fonts using ivfflat (embedding vector_cosine_ops);
```

4. Write a Python script in `/model/scripts/seed_fonts.py` that:
    - Fetches all Google Fonts metadata via the **Google Fonts API** (free, just needs an API key)
    - Inserts font names, categories, and Google Fonts URLs into Supabase
5. Run the seed script — this populates ~1,500 fonts with zero cost

***

## Phase 3: Model Training Pipeline

**Goal:** Train a lightweight font classifier using synthesized data.

1. **Synthesize training data** — create `/model/scripts/generate_dataset.py`:
    - Download all Google Fonts `.ttf` files
    - Use `Pillow` to render 5–10 text samples per font (vary text, size, color, slight rotation/noise)
    - Target: ~10,000–15,000 images across 1,500 classes
2. **Model architecture** — create `/model/train.py`:
    - Use `EfficientNet-B0` as backbone (pre-trained on ImageNet via `timm`)
    - Replace the final classifier head with a `Linear(1280, 1500)` layer for 1,500 font classes
    - Also output a **256-dim embedding** from the penultimate layer for similarity search
3. **Training config:**
    - Optimizer: AdamW, LR: 1e-4
    - Epochs: 30–50
    - Use Google Colab (free T4 GPU) to train — export model as `font_model.pt`
4. **Generate font embeddings** — after training, run inference on a clean sample of each font to generate its 256-dim embedding, then upload all embeddings to Supabase using the seed script

***

## Phase 4: FastAPI Inference Service (Hugging Face Space)

**Goal:** Deploy the model as a REST API endpoint.

1. Create `/model/app.py` as a FastAPI app:

```python
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch, io, numpy as np

app = FastAPI()
model = load_model("font_model.pt")  # load your trained model

@app.get("/ping")
def ping(): return {"status": "awake"}

@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    embedding = extract_embedding(model, img)  # 256-dim vector
    # Return embedding to frontend; Supabase does the vector search
    return {"embedding": embedding.tolist()}
```

2. Create a `Dockerfile` in `/model` to containerize it for Hugging Face Spaces
3. Push `/model` to your Hugging Face Space repo via Git
4. Test the `/ping` and `/identify` endpoints using Postman or `curl`
5. Add your Supabase keys as **HF Space Secrets** (not hardcoded)

***

## Phase 5: Frontend (Next.js on Vercel)

**Goal:** Build the crop UI, call the API, and display results.

### Pages \& Components

- `pages/index.tsx` — main upload + crop + results page
- `components/CropUI.tsx` — wraps `react-image-crop` for text region selection
- `components/FontResult.tsx` — displays font name, category, preview, Google Fonts link
- `components/Uploader.tsx` — drag-and-drop image upload


### Identify Flow (in `pages/api/identify.ts` as a Next.js API route):

```
1. Receive cropped image blob from frontend
2. POST blob → HF Space /identify → get 256-dim embedding back
3. Run pgvector similarity search in Supabase:
   SELECT name, category, google_fonts_url, preview_url,
          1 - (embedding <=> '[vector]') as similarity
   FROM fonts
   ORDER BY similarity DESC
   LIMIT 5;
4. Return top 5 matches to frontend
```


### UX Details

- On page load: call `GET /ping` to wake the HF Space silently
- Show a **crop guide overlay** with instructions ("Draw a box around one line of text")
- Display results as cards: font name, similarity %, category badge, "View on Google Fonts" button
- Add a **"Was this correct?"** thumbs up/down button — store confirmed matches in a `feedback` Supabase table for future training

***

## Phase 6: Deploy \& Polish

**Goal:** Ship it and make it portfolio-ready.

1. Push `/web` to GitHub → connect to **Vercel** → auto-deploys on every push
2. Add a custom domain (optional, free subdomain via Vercel: `fontid.vercel.app`)
3. Set all env vars in Vercel dashboard
4. **Portfolio extras to add:**
    - A "How it works" section explaining the crop → embed → match pipeline
    - Show the tech stack badges (Next.js, Supabase, HF, PyTorch)
    - Link to your GitHub repo so recruiters can read the code
    - Add an example GIF of the tool in action on the landing page
5. Monitor usage via Supabase dashboard (query count) and HF Space logs

***

## Build Order Summary

| Phase | Effort | Dependency |
| :-- | :-- | :-- |
| 1 — Setup | 1–2 hrs | None |
| 2 — Font DB | 2–3 hrs | Phase 1 |
| 3 — Model Training | 1–2 days | Phase 2 |
| 4 — FastAPI on HF | 3–4 hrs | Phase 3 |
| 5 — Frontend | 1–2 days | Phase 4 |
| 6 — Deploy | 1–2 hrs | Phase 5 |

**Total estimated time: 4–6 days** of focused work. Start with Phase 2 and 3 in parallel since the DB setup and dataset generation are independent.

