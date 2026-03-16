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
# Wrap in try-except for local dev where env vars might be missing
supabase = None
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

if supabase_url and supabase_key:
    supabase = create_client(supabase_url, supabase_key)
else:
    print("WARNING: SUPABASE_URL or SUPABASE_SERVICE_KEY missing. Vector search will fail.")

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

        if supabase is None:
             return JSONResponse(content={"matches": [], "local_top5": top5_local, "msg": "Supabase not configured"})

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
                    "preview_url": match.get("preview_url"),
                    "similarity": round(match["similarity"] * 100, 1) if "similarity" in match else None,
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
    if supabase is None:
        raise HTTPException(status_code=503, detail="Supabase not configured")

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
