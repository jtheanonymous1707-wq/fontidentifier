import os, json, time
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# ── Model ─────────────────────────────────────────────────────────────────────
class FontNet(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 256):
        super().__init__()

        # FasterViT-2 — SOTA for font recognition (87.4% top-1)
        self.backbone = timm.create_model(
            'fastervit_2_224',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        backbone_out = self.backbone.num_features  # 768

        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_out, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),              # GELU works better than ReLU for ViT models
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x: torch.Tensor):
        features  = self.backbone(x)
        embedding = self.embedding_head(features)
        embedding = nn.functional.normalize(embedding, p=2.0, dim=1)
        logits    = self.classifier(embedding)
        return logits, embedding


# ── Train Function ─────────────────────────────────────────────────────────────
def train(
    dataset_dir:     str,
    save_dir:        str,
    checkpoint_path: str   = None,
    best_model_path: str   = None,
    epochs:          int   = 60,
    lr:              float = 1e-4,    # lower LR for FasterViT
    batch_size:      int   = 32,      # smaller batch for larger model
    patience:        int   = 10,
    embedding_dim:   int   = 256,
    warmup_epochs:   int   = 5        # LR warmup for stable fine-tuning
):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(save_dir, "checkpoints", "latest.pt")
    if best_model_path is None:
        best_model_path = os.path.join(save_dir, "checkpoints", "best_model.pt")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(best_model_path),  exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device     : {device}")
    if torch.cuda.is_available():
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
        print(f"VRAM       : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # ── Transforms ────────────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ── Dataset ────────────────────────────────────────────────────────────────
    full_dataset = datasets.ImageFolder(dataset_dir)
    num_classes  = len(full_dataset.classes)
    print(f"Classes    : {num_classes:,}")
    print(f"Images     : {len(full_dataset):,}")

    class_index = {v: k for k, v in full_dataset.class_to_idx.items()}
    with open(os.path.join(save_dir, "class_index.json"), "w") as f:
        json.dump(class_index, f)
    print("class_index.json saved ✅")

    val_size   = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_set.dataset.transform = train_transform
    val_set.dataset.transform   = val_transform

    # num_workers=4 for faster data loading with larger images
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model     = FontNet(num_classes=num_classes, embedding_dim=embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Layerwise LR — backbone gets 10x lower LR than heads (standard fine-tuning)
    optimizer = AdamW([
        {"params": model.backbone.parameters(),       "lr": lr * 0.1},
        {"params": model.embedding_head.parameters(), "lr": lr},
        {"params": model.classifier.parameters(),     "lr": lr},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    # ── Mixed Precision Scaler (speeds up training ~2x on T4) ─────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # ── Resume ─────────────────────────────────────────────────────────────────
    start_epoch    = 0
    best_val_acc   = 0.0
    patience_count = 0

    if os.path.exists(checkpoint_path):
        print(f"\nCheckpoint found — resuming...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        scaler.load_state_dict(ckpt['scaler_state'])
        start_epoch    = ckpt['epoch'] + 1
        best_val_acc   = ckpt['best_val_acc']
        patience_count = ckpt['patience_count']
        print(f"Resumed from epoch {start_epoch} | Best: {best_val_acc:.4f}")
    else:
        print(f"\nStarting fresh training with FasterViT-2")

    print("─" * 60)

    # ── Training Loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # Warmup LR for first N epochs — prevents instability in fine-tuning
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = pg['lr'] * warmup_factor

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = correct = total = 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits, _ = model(imgs)
                loss      = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

            if batch_idx % 30 == 0:
                print(
                    f"  E{epoch+1:03d} | "
                    f"Batch {batch_idx:4d}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Acc: {correct/total:.4f}",
                    end='\r'
                )

        if epoch >= warmup_epochs:
            scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_correct = top5_correct = val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits, _ = model(imgs)

                # Top-1
                val_correct += (logits.argmax(1) == labels).sum().item()
                # Top-5
                top5        = logits.topk(5, dim=1).indices
                top5_correct += sum(
                    labels[i].item() in top5[i].tolist()
                    for i in range(labels.size(0))
                )
                val_total += labels.size(0)

        train_acc = correct / total
        val_acc   = val_correct / val_total
        top5_acc  = top5_correct / val_total
        elapsed   = time.time() - epoch_start
        current_lr = optimizer.param_groups[1]['lr']  # head LR

        print(
            f"\nEpoch {epoch+1:03d}/{epochs} | "
            f"Train: {train_acc:.4f} | "
            f"Val@1: {val_acc:.4f} | "
            f"Val@5: {top5_acc:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.0f}s"
        )

        # ── Save checkpoint every epoch ────────────────────────────────────────
        torch.save({
            'epoch':           epoch,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state':    scaler.state_dict(),
            'best_val_acc':    best_val_acc,
            'patience_count':  patience_count,
            'num_classes':     num_classes,
            'val_acc':         val_acc,
            'top5_acc':        top5_acc,
        }, checkpoint_path)

        # ── Save best model ────────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'num_classes': num_classes,
                'val_acc':     val_acc,
                'top5_acc':    top5_acc,
            }, best_model_path)
            print(
                f"  ✅ Best model saved | "
                f"Top-1: {val_acc:.4f} | Top-5: {top5_acc:.4f}"
            )
        else:
            patience_count += 1
            print(f"  No improvement ({patience_count}/{patience})")
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("\n" + "─" * 60)
    print(f"Training complete.")
    print(f"Best Top-1 val accuracy : {best_val_acc:.4f}")
