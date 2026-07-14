"""
Run: pytest tests/
"""
import torch

from src.losses import DiceBCELoss
from src.metrics import dice_coefficient, iou_score
from src.model import AttentionUNet


def test_model_output_shape():
    model = AttentionUNet(img_size=256, pretrained=False)
    x = torch.zeros(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 256, 256)


def test_dice_bce_loss_is_low_for_perfect_prediction():
    loss_fn = DiceBCELoss()
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 2:5, 2:5] = 1.0
    # logits that saturate sigmoid to (near) match target exactly
    logits = (target * 20) - 10
    loss = loss_fn(logits, target)
    assert loss.item() < 0.05


def test_dice_bce_loss_is_high_for_inverted_prediction():
    loss_fn = DiceBCELoss()
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 2:5, 2:5] = 1.0
    inverted_logits = ((1 - target) * 20) - 10
    loss = loss_fn(inverted_logits, target)
    assert loss.item() > 0.9


def test_dice_coefficient_perfect_overlap_is_one():
    mask = torch.zeros(1, 4, 4)
    mask[0, 1:3, 1:3] = 1.0
    assert dice_coefficient(mask, mask) > 0.99


def test_dice_coefficient_no_overlap_is_near_zero():
    # Use masks large enough that the smoothing constant (added to avoid
    # divide-by-zero on empty masks) doesn't dominate the result -- with
    # single-pixel masks the smoothing term alone gives a misleadingly
    # high score, which this test caught during development.
    pred = torch.zeros(1, 16, 16)
    pred[0, 0:4, 0:4] = 1.0
    target = torch.zeros(1, 16, 16)
    target[0, 12:16, 12:16] = 1.0
    assert dice_coefficient(pred, target) < 0.1


def test_iou_is_never_greater_than_dice():
    # Mathematical property: IoU <= Dice for the same pair of masks, always.
    torch.manual_seed(0)
    pred = torch.rand(1, 16, 16)
    target = (torch.rand(1, 16, 16) > 0.5).float()
    assert iou_score(pred, target) <= dice_coefficient(pred, target) + 1e-6
