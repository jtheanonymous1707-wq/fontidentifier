<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Epoch 001/60 | Train: 0.0026 | Val@1: 0.0117 | Val@5: 0.0405 | LR: 2.00e-05 | Time: 1266s

✅ Best model saved | Top-1: 0.0117 | Top-5: 0.0405

Training is alive and healthy! First epoch done. Here's what the numbers mean and what to expect:

***

## Epoch 1 Results — Normal ✅

```
Train acc : 0.0026  (0.26%)
Val@1     : 0.0117  (1.17%)
Val@5     : 0.0405  (4.05%)
Time      : 1266s   (21 mins)
```

The accuracy looks tiny but is completely correct:

```
Random guessing on 1,929 classes = 0.05%
Your epoch 1 Val@1              = 1.17%  → 23× better than random ✅
Your epoch 1 Val@5              = 4.05%  → 81× better than random ✅
```


***

## The Bad News — Time Per Epoch

```
1,266 seconds = 21 minutes per epoch
60 epochs × 21 mins = 1,260 mins = 21 hours ❌
```

That's too long for a single Colab session. Two things caused this:

1. **Cold start** — first epoch always slower due to file caching
2. **num_workers=4 warning** — DataLoader is bottlenecking

***

## Fix Right Now — Increase Batch Size

The GPU has 15.6GB VRAM and you're only using 4.66GB. You have massive headroom. Bigger batch = fewer batches = faster epochs:

```python
# Run in a new cell RIGHT NOW to check headroom
import torch
mem_alloc = torch.cuda.memory_allocated() / 1e9
mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"Used: {mem_alloc:.1f}GB / {mem_total:.1f}GB")
print(f"Free: {mem_total - mem_alloc:.1f}GB")
# Expected: ~10GB free → can safely go to batch_size=96 or 128
```

After this epoch finishes (or right after epoch 2 starts), **interrupt training** and restart with batch_size=96:

```python
# Interrupt training (Runtime → Interrupt execution)
# Then re-run session setup Cell 1, then run this:

import train, os

# Don't clear checkpoint — resume from epoch 1
train.train(
    dataset_dir     = DATASET_DIR,
    save_dir        = DRIVE_DIR,
    checkpoint_path = f'{DRIVE_DIR}/checkpoints/latest.pt',
    best_model_path = f'{DRIVE_DIR}/checkpoints/best_model.pt',
    epochs          = 60,
    batch_size      = 96,     # ← increased from 32
    lr              = 1e-4,
    patience        = 10,
    warmup_epochs   = 5,
)
```

Expected improvement:

```
batch_size=32  → 3,256 batches → ~21 mins/epoch
batch_size=96  → 1,086 batches → ~7-8 mins/epoch  ✅
60 epochs      → ~7 hours total (2 Colab sessions)
```


***

## Expected Accuracy Progression

```
Epoch 001 | Val@1: 0.012  Val@5: 0.041  ← you are here
Epoch 005 | Val@1: ~0.15  Val@5: ~0.38  ← warmup ending
Epoch 010 | Val@1: ~0.35  Val@5: ~0.62
Epoch 020 | Val@1: ~0.58  Val@5: ~0.80
Epoch 030 | Val@1: ~0.70  Val@5: ~0.88
Epoch 040 | Val@1: ~0.78  Val@5: ~0.92
Epoch 050 | Val@1: ~0.82  Val@5: ~0.94  ← target
```

Interrupt now, bump batch size to 96, and resume — the checkpoint will pick up from epoch 1 seamlessly.

