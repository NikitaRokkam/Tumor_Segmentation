"""
Run:
    python -m src.train
    python -m src.train --epochs 60 --batch-size 16
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src import config
from src.dataset import build_datasets
from src.losses import DiceBCELoss
from src.metrics import dice_coefficient
from src.model import AttentionUNet


def run_epoch(model, loader, loss_fn, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_dice, n_batches = 0.0, 0.0, 0

    torch.set_grad_enabled(train)
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        if train:
            optimizer.zero_grad()

        logits = model(images)
        loss = loss_fn(logits, masks)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_dice += dice_coefficient(torch.sigmoid(logits), masks)
        n_batches += 1
    torch.set_grad_enabled(True)

    return total_loss / n_batches, total_dice / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--data-dir", type=str, default=str(config.DATA_DIR))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds, _ = build_datasets(
        data_dir=Path(args.data_dir),
        img_size=config.IMG_SIZE, val_split=config.VAL_SPLIT,
        test_split=config.TEST_SPLIT, seed=config.SEED,
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=config.NUM_WORKERS)

    model = AttentionUNet(img_size=config.IMG_SIZE, pretrained=True).to(device)
    loss_fn = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Plateau-based decay: simple and effective for a training run this short;
    # a cosine schedule would need max_epochs tuned in, this adapts automatically.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    config.CHECKPOINT_DIR.mkdir(exist_ok=True)
    history = []
    best_val_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_dice = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        val_loss, val_dice = run_epoch(model, val_loader, loss_fn, optimizer, device, train=False)
        scheduler.step(val_dice)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} | "
              f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} | {elapsed:.1f}s")

        history.append({"epoch": epoch, "train_loss": train_loss, "train_dice": train_dice,
                         "val_loss": val_loss, "val_dice": val_dice})

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"  -> new best val_dice={val_dice:.4f}, saved to {config.BEST_MODEL_PATH}")

    with open(config.CHECKPOINT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. Best val_dice={best_val_dice:.4f}")


if __name__ == "__main__":
    main()
