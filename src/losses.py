"""
Hybrid Dice + BCE loss.

"""
import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, target)

        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (probs_flat * target_flat).sum(dim=1)
        dice_coeff = (2.0 * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )
        dice_loss = 1.0 - dice_coeff.mean()

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss
