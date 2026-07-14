
import torch


@torch.no_grad()
def dice_coefficient(probs: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1.0) -> float:
    """Overlap metric, matches the Dice loss term -- the primary metric we optimize for and report."""
    preds = (probs > threshold).float()
    intersection = (preds * target).sum()
    return ((2.0 * intersection + smooth) / (preds.sum() + target.sum() + smooth)).item()


@torch.no_grad()
def iou_score(probs: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1.0) -> float:
    """
    Intersection-over-Union. Stricter than Dice for the same prediction (IoU < Dice
    always, for the same masks) -- useful as a secondary check since a model can
    look good on Dice while its boundaries are still noticeably off.
    """
    preds = (probs > threshold).float()
    intersection = (preds * target).sum()
    union = preds.sum() + target.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


@torch.no_grad()
def pixel_accuracy(probs: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Included as a sanity check only -- NOT a metric to optimize for or report as
    primary. Because background pixels dominate, a model that predicts all-zero
    still scores 90%+ pixel accuracy on most images, which is misleading on its own.
    """
    preds = (probs > threshold).float()
    correct = (preds == target).float().sum()
    return (correct / target.numel()).item()
