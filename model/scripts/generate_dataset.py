import os, random, glob
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np

FONTS_DIR = "data/fonts"
OUTPUT_DIR = "data/dataset"
IMG_SIZE = (128, 64)
SAMPLES_PER_FONT = 50
SAMPLE_TEXTS = [
    "Handgloves", "Typography", "AaBbCc", "FontID",
    "Design 2024", "Hello World", "Quick fox", "Spectrum",
    "ABCDE fghij", "Waltz nymph", "Bright vixens",
    "Pack my box", "Sphinx of black", "Five boxing",
    "Jackdaws love", "Quartz glyph"
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
        x = max(0, (IMG_SIZE[0] - (bbox[2] - bbox[0])) // 2)
        y = max(0, (IMG_SIZE[1] - (bbox[3] - bbox[1])) // 2)
        color = (random.randint(0, 60),) * 3
        draw.text((x, y), text, font=font, fill=color)
        return img
    except:
        return None

def generate_dataset():
    font_paths = glob.glob(os.path.join(FONTS_DIR, '*.*'))
    print(f"Generating dataset for {len(font_paths)} fonts...")
    skipped, generated = 0, 0

    for i, font_path in enumerate(font_paths):
        font_name = os.path.splitext(os.path.basename(font_path))[0]
        class_dir = os.path.join(OUTPUT_DIR, font_name)
        os.makedirs(class_dir, exist_ok=True)

        # ← Resume: skip if already fully generated
        existing = len(os.listdir(class_dir))
        if existing >= SAMPLES_PER_FONT:
            skipped += 1
            continue

        count = 0
        for text in SAMPLE_TEXTS:
            for font_size in [24, 28, 32, 36, 40, 44]:
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

    print(f"Dataset ready at {OUTPUT_DIR}")

    print("Removing empty class folders...")
    removed = 0
    for class_dir in os.listdir(OUTPUT_DIR):
        full_path = os.path.join(OUTPUT_DIR, class_dir)
        if os.path.isdir(full_path) and len(os.listdir(full_path)) == 0:
            os.rmdir(full_path)
            removed += 1
    print(f"Cleaned {removed} empty folders. Dataset ready.")

if __name__ == "__main__":
    generate_dataset()
