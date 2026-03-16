import torch
import os

MODEL_PATH = "font_model_scripted.pt"

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: {MODEL_PATH} not found.")
    exit(1)

try:
    print(f"Attempting to load {MODEL_PATH}...")
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    print("SUCCESS: Model loaded and set to eval mode.")
    
    # Check if we can do a dummy inference
    # The preprocessor uses (64, 128)
    dummy_input = torch.randn(1, 3, 64, 128)
    with torch.no_grad():
        output = model(dummy_input)
    print("SUCCESS: Dummy inference successful.")
    if isinstance(output, tuple):
        print(f"Output is a tuple of length {len(output)}")
        for i, o in enumerate(output):
            print(f"  Part {i} shape: {o.shape}")
    else:
        print(f"Output shape: {output.shape}")
        
except Exception as e:
    print(f"FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
