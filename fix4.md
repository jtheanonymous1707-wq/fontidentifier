<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Using device: cuda


---------------------------------------------------------------------------

FileNotFoundError                         Traceback (most recent call last)

[/tmp/ipykernel_9488/783220733.py](https://localhost:8080/#) in <cell line: 0>()
1 import train
----> 2 train.train(
3     dataset_dir     = '/content/data/dataset', \# Path in Colab
4     save_dir        = DRIVE_DIR,
5     checkpoint_path = f'{DRIVE_DIR}/checkpoints/latest.pt',

4 frames
[/usr/local/lib/python3.12/dist-packages/torchvision/datasets/folder.py](https://localhost:8080/#) in make_dataset(directory, class_to_idx, extensions, is_valid_file, allow_empty)
102         if extensions is not None:
103             msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
--> 104         raise FileNotFoundError(msg)
105
106     return instances

FileNotFoundError: Found no valid file for the classes Kumar_One, Kumar_One_Outline. Supported extensions are: .jpg, .jpeg, .png, .ppm, .bmp, .pgm, .tif, .tiff, .webp

Two problems happening here — the dataset either wasn't extracted to Colab, or some font folders were created empty during generation. Here's how to fix both:

***

## Problem 1 — Dataset Not Extracted in Colab

The `/content/data/dataset` path is empty or missing. Add this **extraction cell** before calling `train.train()`:

```python
import zipfile, os, glob

DATASET_ZIP = f'{DRIVE_DIR}/dataset.zip'
DATASET_DIR = '/content/data/dataset'

# Only extract if not already there
if not os.path.exists(DATASET_DIR) or len(glob.glob(f'{DATASET_DIR}/*')) == 0:
    print("Extracting dataset from Drive...")
    with zipfile.ZipFile(DATASET_ZIP, 'r') as z:
        z.extractall('/content/')
    print("Extraction complete.")
else:
    print(f"Dataset already extracted: {len(os.listdir(DATASET_DIR))} classes found")
```


***

## Problem 2 — Empty Font Folders (Kumar_One etc.)

Some fonts render no valid images because the `.ttf` file exists but the font doesn't support the sample text characters. `torchvision.ImageFolder` crashes when it finds an empty class folder.

Fix it by **purging all empty folders** right after extraction:

```python
import shutil

print("Scanning for empty class folders...")
removed = 0
for class_dir in os.listdir(DATASET_DIR):
    full_path = os.path.join(DATASET_DIR, class_dir)
    images = glob.glob(f'{full_path}/*.png')
    if len(images) == 0:
        shutil.rmtree(full_path)
        removed += 1

total_classes = len(os.listdir(DATASET_DIR))
total_images  = len(glob.glob(f'{DATASET_DIR}/*/*.png'))
print(f"Removed {removed} empty folders")
print(f"Clean dataset: {total_images} images across {total_classes} classes")
```


***

## Also Fix `generate_dataset.py` Locally

To prevent this from happening again when you regenerate with 50 samples, add an empty folder guard to your local script:

```python
# Add this at the END of the generate_dataset() function
# in model/scripts/generate_dataset.py

print("Removing empty class folders...")
removed = 0
for class_dir in os.listdir(OUTPUT_DIR):
    full_path = os.path.join(OUTPUT_DIR, class_dir)
    if len(os.listdir(full_path)) == 0:
        os.rmdir(full_path)
        removed += 1
print(f"Cleaned {removed} empty folders. Dataset ready.")
```


***

## Full Correct Cell Order for This Session

```python
# Step 1 — Extract dataset
import zipfile, os, glob, shutil

DATASET_DIR = '/content/data/dataset'

if not os.path.exists(DATASET_DIR) or len(glob.glob(f'{DATASET_DIR}/*')) == 0:
    print("Extracting dataset...")
    with zipfile.ZipFile(f'{DRIVE_DIR}/dataset.zip', 'r') as z:
        z.extractall('/content/')

# Step 2 — Remove empty folders
removed = 0
for class_dir in os.listdir(DATASET_DIR):
    full_path = os.path.join(DATASET_DIR, class_dir)
    if len(glob.glob(f'{full_path}/*.png')) == 0:
        shutil.rmtree(full_path)
        removed += 1

total_classes = len(os.listdir(DATASET_DIR))
total_images  = len(glob.glob(f'{DATASET_DIR}/*/*.png'))
print(f"Removed {removed} empty folders")
print(f"Ready: {total_images} images | {total_classes} classes")

# Step 3 — Train
import train
train.train(
    dataset_dir     = DATASET_DIR,
    save_dir        = DRIVE_DIR,
    checkpoint_path = f'{DRIVE_DIR}/checkpoints/latest.pt',
    best_model_path = f'{DRIVE_DIR}/checkpoints/best_model.pt',
    epochs          = 80,
    batch_size      = 128,
    lr              = 3e-4,
    patience        = 12,
)
```

