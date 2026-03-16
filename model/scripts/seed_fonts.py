import os, requests
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# Check for environment variables
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
google_api_key = os.environ.get("GOOGLE_FONTS_API_KEY")

if not all([supabase_url, supabase_key, google_api_key]):
    print("Error: Missing environment variables. Please check SUPABASE_URL, SUPABASE_SERVICE_KEY, and GOOGLE_FONTS_API_KEY.")
    exit(1)

supabase = create_client(supabase_url, supabase_key)

def fetch_google_fonts():
    url = f"https://www.googleapis.com/webfonts/v1/webfonts?key={google_api_key}&sort=alpha"
    res = requests.get(url)
    res.raise_for_status()
    return res.json()["items"]

def seed():
    try:
        fonts = fetch_google_fonts()
        print(f"Fetched {len(fonts)} fonts from Google Fonts API")
    except Exception as e:
        print(f"Failed to fetch fonts: {e}")
        return

    batch = []
    for font in fonts:
        name = font["family"]
        category = font.get("category", "sans-serif")
        variants = font.get("variants", [])
        subsets = font.get("subsets", [])
        google_fonts_url = f"https://fonts.google.com/specimen/{name.replace(' ', '+')}"

        batch.append({
            "name": name,
            "family": name,
            "category": category,
            "variants": variants,
            "subsets": subsets,
            "google_fonts_url": google_fonts_url,
            "preview_url": None,   # filled later
            "embedding": None      # filled in Phase 3
        })

    # Insert in batches of 100 to avoid timeouts
    for i in range(0, len(batch), 100):
        chunk = batch[i:i+100]
        try:
            supabase.table("fonts").insert(chunk).execute()
            print(f"Inserted fonts {i+1} to {i+len(chunk)}")
        except Exception as e:
            print(f"Error inserting chunk {i+1}-{i+len(chunk)}: {e}")

    print("Seeding complete!")

if __name__ == "__main__":
    seed()
