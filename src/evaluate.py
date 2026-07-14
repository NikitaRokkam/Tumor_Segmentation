"""
Run:
    python -m src.evaluate
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config
from src.dataset import build_datasets
from src.metrics import dice_coefficient, iou_score, pixel_accuracy
from src.model import AttentionUNet


def denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (image_tensor.cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def save_example_grid(images, masks, preds, out_path, n_examples: int = 6):
    n_examples = min(n_examples, len(images))
    fig, axes = plt.subplots(n_examples, 3, figsize=(9, 3 * n_examples))
    for i in range(n_examples):
        axes[i, 0].imshow(denormalize(images[i]))
        axes[i, 0].set_title("Input")
        axes[i, 1].imshow(masks[i, 0].cpu(), cmap="gray")
        axes[i, 1].set_title("Ground truth")
        axes[i, 2].imshow(preds[i, 0].cpu(), cmap="gray")
        axes[i, 2].set_title("Prediction")
        for ax in axes[i]:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"Saved example predictions to {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_ds = build_datasets(
        data_dir=config.DATA_DIR, img_size=config.IMG_SIZE,
        val_split=config.VAL_SPLIT, test_split=config.TEST_SPLIT, seed=config.SEED,
    )
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    print(f"Test set size: {len(test_ds)}")

    model = AttentionUNet(img_size=config.IMG_SIZE, pretrained=False).to(device)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    dice_scores, iou_scores, acc_scores = [], [], []
    example_images, example_masks, example_preds = [], [], []

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            probs = torch.sigmoid(model(images))

            for i in range(images.size(0)):
                dice_scores.append(dice_coefficient(probs[i], masks[i], config.MASK_THRESHOLD))
                iou_scores.append(iou_score(probs[i], masks[i], config.MASK_THRESHOLD))
                acc_scores.append(pixel_accuracy(probs[i], masks[i], config.MASK_THRESHOLD))

            if len(example_images) < 6:
                preds_binary = (probs > config.MASK_THRESHOLD).float()
                example_images.append(images)
                example_masks.append(masks)
                example_preds.append(preds_binary)

    print("\n=== Test set results ===")
    print(f"Dice coefficient : {np.mean(dice_scores):.4f} (+/- {np.std(dice_scores):.4f})")
    print(f"IoU              : {np.mean(iou_scores):.4f} (+/- {np.std(iou_scores):.4f})")
    print(f"Pixel accuracy   : {np.mean(acc_scores):.4f}  (sanity check only, see metrics.py docstring)")

    worst_idx = np.argsort(dice_scores)[:5]
    print(f"\nWorst 5 Dice scores (inspect these): {[round(dice_scores[i], 3) for i in worst_idx]}")

    config.RESULTS_DIR.mkdir(exist_ok=True)
    images_cat = torch.cat(example_images)
    masks_cat = torch.cat(example_masks)
    preds_cat = torch.cat(example_preds)
    save_example_grid(images_cat, masks_cat, preds_cat, config.RESULTS_DIR / "test_predictions.png")


if __name__ == "__main__":
    main()
