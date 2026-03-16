import os, requests, zipfile, io, glob
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_FONTS_API_KEY")
FONTS_DIR = "data/fonts"
os.makedirs(FONTS_DIR, exist_ok=True)

# ── Google Fonts ─────────────────────────────────────────
def download_google_fonts():
    if not API_KEY:
        print("Warning: GOOGLE_FONTS_API_KEY not found. Skipping Google Fonts.")
        return
    
    res = requests.get(f"https://www.googleapis.com/webfonts/v1/webfonts?key={API_KEY}&sort=alpha")
    if res.status_code != 200:
        print(f"Error fetching Google Fonts: {res.text}")
        return
        
    fonts = res.json().get('items', [])
    print(f"Found {len(fonts)} Google Fonts")

    for font in fonts:
        name = font['family'].replace(' ', '_')
        save_path = os.path.join(FONTS_DIR, f"{name}.ttf")
        if os.path.exists(save_path):
            continue
        
        files = font.get('files', {})
        url = files.get('regular') or list(files.values())[0] if files else None
        if not url:
            continue
            
        try:
            r = requests.get(url.replace('http://', 'https://'), timeout=10)
            with open(save_path, 'wb') as f:
                f.write(r.content)
            print(f"✓ Downloaded {name} (Google)")
        except Exception as e:
            print(f"  Skipped {name}: {e}")

# ── Font Squirrel ─────────────────────────────────────────
def download_fontsquirrel(limit=2000):
    print("Fetching Font Squirrel list...")
    try:
        res = requests.get("https://www.fontsquirrel.com/api/fontlist/all", timeout=30)
        fonts = res.json()
    except Exception as e:
        print(f"Error fetching Font Squirrel list: {e}")
        return

    print(f"Found {len(fonts)} Font Squirrel fonts. Capping at {limit} for now.")
    
    downloaded_count = 0
    for font in fonts:
        if downloaded_count >= limit:
            break
            
        slug = font['family_urlname']
        name = slug.replace('-', '_')
        save_path = os.path.join(FONTS_DIR, f"{name}.ttf")
        
        if os.path.exists(save_path):
            downloaded_count += 1
            continue
            
        try:
            url = f"https://www.fontsquirrel.com/fonts/download/{slug}"
            r = requests.get(url, timeout=20)
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                # Extract only a regular weight TTF/OTF
                for fname in z.namelist():
                    fname_lower = fname.lower()
                    is_regular = any(w in fname_lower for w in ['regular', '-400', 'book', 'roman'])
                    is_font = fname_lower.endswith('.ttf') or fname_lower.endswith('.otf')
                    
                    if is_font and (is_regular or not any(
                        w in fname_lower for w in ['bold','italic','light','thin','black','medium']
                    )):
                        with open(save_path, 'wb') as f:
                            f.write(z.read(fname))
                        print(f"✓ Downloaded {name} (FS)")
                        downloaded_count += 1
                        break
        except Exception as e:
            pass # Skip failed downloads

if __name__ == "__main__":
    download_google_fonts()
    download_fontsquirrel()
    
    total = len(glob.glob(os.path.join(FONTS_DIR, '*.*')))
    print(f"\nTotal fonts in {FONTS_DIR}: {total}")
