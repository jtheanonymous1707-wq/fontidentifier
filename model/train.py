import os, json
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ── Model Architecture ──────────────────────────────────────
class FontNet(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 256):
        super().__init__()
        # EfficientNet-B0 pretrained backbone
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0,        # Remove default classifier head
            global_pool='avg'
        )
        backbone_out = self.backbone.num_features  # 1280 for B0

        # Embedding head — for similarity search in Supabase
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_out, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)   # L2-normalize later
        )

        # Classification head — for training supervision
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x, return_embedding=False):
        features = self.backbone(x)
        embedding = self.embedding_head(features)
        # L2 normalize the embedding
        embedding = nn.functional.normalize(embedding, p=2, dim=1)

        if return_embedding:
            return embedding

        logits = self.classifier(embedding)
        return logits, embedding

# ── Config ──────────────────────────────────────────────────
DATASET_DIR  = "data/dataset"
DRIVE_DIR    = "checkpoints" # Default for local, updated in Colab
CHECKPOINT_PATH = os.path.join(DRIVE_DIR, "latest.pt")
BEST_MODEL_PATH = os.path.join(DRIVE_DIR, "best_model.pt")
BATCH_SIZE   = 64
EPOCHS       = 60
LR           = 3e-4
EMBEDDING_DIM = 256
PATIENCE     = 8     # early stopping

# ── Transforms ──────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((64, 128)),
    transforms.RandomHorizontalFlip(p=0.1),    # only subtle for text
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])  # ImageNet stats
])

val_transform = transforms.Compose([
    transforms.Resize((64, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # ── Dataset Split ────────────────────────────────────────────
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory {DATASET_DIR} not found. Generate it first.")
        return

    full_dataset = datasets.ImageFolder(DATASET_DIR)
    num_classes  = len(full_dataset.classes)

    # Save class index mapping to JSON (needed for inference later)
    with open(os.path.join(SAVE_DIR, "class_index.json"), "w") as f:
        json.dump({v: k for k, v in full_dataset.class_to_idx.items()}, f)

    val_size   = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    # Apply transforms by wrapping the split datasets
    train_set.dataset.transform = train_transform
    val_set.dataset.transform   = val_transform

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── Model, Loss, Optimizer ───────────────────────────────────
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = FontNet(num_classes=num_classes, embedding_dim=EMBEDDING_DIM).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    # ── Auto-resume from checkpoint ───────────────────────────
    start_epoch    = 0
    best_val_acc   = 0.0
    patience_count = 0

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint found at {CHECKPOINT_PATH} — resuming training...")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch    = ckpt['epoch'] + 1
        best_val_acc   = ckpt['best_val_acc']
        patience_count = ckpt['patience_count']
        print(f"Resumed from epoch {start_epoch} | Best val acc so far: {best_val_acc:.4f}")
    else:
        print(f"Starting fresh training | {num_classes} classes | {len(full_dataset)} images")

    # ── Training Loop ────────────────────────────────────────────
    print(f"Starting training on {device}...")

    for epoch in range(start_epoch, EPOCHS):
        # — Train —
        model.train()
        train_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
            optimizer.step()
            train_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)
        scheduler.step()

        # — Validate —
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, _ = model(imgs)
                val_loss    += criterion(logits, labels).item()
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        train_acc = correct / total
        val_acc   = val_correct / val_total
        print(f"Epoch {epoch+1:03d}/{EPOCHS} | "
              f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # ── Save latest checkpoint every epoch (resume safety) ──
        torch.save({
            'epoch':           epoch,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_val_acc':    best_val_acc,
            'patience_count':  patience_count,
            'num_classes':     num_classes,
            'val_acc':         val_acc
        }, CHECKPOINT_PATH)

        # ── Save best model separately ──────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_count = 0
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'num_classes': num_classes,
                'val_acc':     val_acc
            }, BEST_MODEL_PATH)
            print(f"  ✅ New best model saved (val_acc={val_acc:.4f})")
        else:
            patience_count += 1
            print(f"  No improvement ({patience_count}/{PATIENCE})")
            if patience_count >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

if __name__ == "__main__":
    train()
