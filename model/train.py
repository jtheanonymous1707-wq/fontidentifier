import os, json, time
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ── Model Definition ─────────────────────────────────────────────────────────
class FontNet(nn.Module):
    def __init__(self, num_classes, embedding_dim=256):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b0', pretrained=True, num_classes=0, global_pool='avg'
        )
        backbone_out = self.backbone.num_features  # 1280

        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_out, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x, return_embedding=False):
        features  = self.backbone(x)
        embedding = self.embedding_head(features)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        if return_embedding:
            return embedding
        return self.classifier(embedding), embedding


# ── Train Function ────────────────────────────────────────────────────────────
def train(
    dataset_dir:      str,
    save_dir:         str,
    checkpoint_path:  str = None,
    best_model_path:  str = None,
    epochs:           int = 60,
    lr:             float = 3e-4,
    batch_size:       int = 64,
    patience:         int = 8,
    embedding_dim:    int = 256
):
    # Default paths derived from save_dir if not explicitly passed
    if checkpoint_path is None:
        checkpoint_path = os.path.join(save_dir, "checkpoints", "latest.pt")
    if best_model_path is None:
        best_model_path = os.path.join(save_dir, "checkpoints", "best_model.pt")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(best_model_path),  exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Transforms ───────────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize((64, 128)),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((64, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ── Dataset ───────────────────────────────────────────────────────────────
    full_dataset = datasets.ImageFolder(dataset_dir)
    num_classes  = len(full_dataset.classes)
    print(f"Classes: {num_classes} | Total images: {len(full_dataset)}")

    # Save class index map
    class_index = {v: k for k, v in full_dataset.class_to_idx.items()}
    with open(os.path.join(save_dir, "class_index.json"), "w") as f:
        json.dump(class_index, f)
    print("class_index.json saved")

    val_size   = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_set.dataset.transform = train_transform
    val_set.dataset.transform   = val_transform

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── Model, Loss, Optimizer ────────────────────────────────────────────────
    model     = FontNet(num_classes=num_classes, embedding_dim=embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    # ── Resume from checkpoint if exists ─────────────────────────────────────
    start_epoch    = 0
    best_val_acc   = 0.0
    patience_count = 0

    if os.path.exists(checkpoint_path):
        print(f"Checkpoint found — resuming from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch    = ckpt['epoch'] + 1
        best_val_acc   = ckpt['best_val_acc']
        patience_count = ckpt['patience_count']
        print(f"Resumed from epoch {start_epoch} | Best val acc: {best_val_acc:.4f}")

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f}", end='\r')

        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, _    = model(imgs)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        train_acc = correct / total
        val_acc   = val_correct / val_total
        elapsed   = time.time() - epoch_start

        print(f"\nEpoch {epoch+1:03d}/{epochs} | "
              f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Time: {elapsed:.0f}s")

        # Save latest checkpoint every epoch
        torch.save({
            'epoch':           epoch,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_val_acc':    best_val_acc,
            'patience_count':  patience_count,
            'num_classes':     num_classes,
            'val_acc':         val_acc
        }, checkpoint_path)

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'num_classes': num_classes,
                'val_acc':     val_acc
            }, best_model_path)
            print(f"  ✅ New best model saved (val_acc={val_acc:.4f})")
        else:
            patience_count += 1
            print(f"  No improvement ({patience_count}/{patience})")
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
