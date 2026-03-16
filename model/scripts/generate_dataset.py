import os, io, random, glob, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

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
