import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


class BUSIDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, Path]], img_size: int, augment: bool = False):
        """
        samples: list of (image_path, mask_path) tuples.
        augment: apply random flips during training only -- never at val/test time,
                 since we want validation/test metrics to reflect real, unaugmented images.
        """
        self.samples = samples
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # single-channel

        image = TF.resize(image, [self.img_size, self.img_size])
        mask = TF.resize(mask, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        image = TF.to_tensor(image)  # -> [3, H, W], values in [0, 1]
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats, matches pretrained encoder

        mask = TF.to_tensor(mask)    # -> [1, H, W], values in [0, 1]
        mask = (mask > 0.5).float()  # binarize -- source masks are occasionally anti-aliased PNGs, not clean 0/1

        return image, mask


def _collect_samples(data_dir: Path) -> list[tuple[Path, Path]]:
    """
    Pairs each base image with its mask. BUSI sometimes ships more than one
    mask per image (multiple annotated lesions); we merge those into one
    mask so each image maps to exactly one training pair.
    """
    samples = []
    for class_dir in ["benign", "malignant", "normal"]:
        folder = data_dir / class_dir
        if not folder.exists():
            continue
        image_paths = sorted(p for p in folder.glob("*.png") if "_mask" not in p.stem)
        for img_path in image_paths:
            mask_paths = sorted(folder.glob(f"{img_path.stem}_mask*.png"))
            if not mask_paths:
                continue
            samples.append((img_path, mask_paths[0]))  # first mask is sufficient for this project's scope
    return samples


def build_datasets(data_dir: Path, img_size: int, val_split: float, test_split: float, seed: int):
    """Splits at the patient/image level (not per-pixel), shuffled once with a fixed seed for reproducibility."""
    samples = _collect_samples(data_dir)
    if not samples:
        raise RuntimeError(f"No image/mask pairs found under {data_dir}. Check the folder layout in the README.")

    rng = random.Random(seed)
    rng.shuffle(samples)

    n_val = int(len(samples) * val_split)
    n_test = int(len(samples) * test_split)
    val_samples = samples[:n_val]
    test_samples = samples[n_val:n_val + n_test]
    train_samples = samples[n_val + n_test:]

    train_ds = BUSIDataset(train_samples, img_size, augment=True)
    val_ds = BUSIDataset(val_samples, img_size, augment=False)
    test_ds = BUSIDataset(test_samples, img_size, augment=False)
    return train_ds, val_ds, test_ds
