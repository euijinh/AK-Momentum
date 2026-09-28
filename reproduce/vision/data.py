"""
CIFAR-10 loaders with the standard augmentation used for every vision run.

Training: random 32x32 crop with 4-pixel padding, random horizontal flip, then
per-channel normalization.  Evaluation: normalization only.  TorchVision
downloads the dataset on first use.
"""

from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def test_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def build_dataloaders(
    data_root: str = "./data/cifar10",
    batch_size: int = 256,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Return ``(train_loader, test_loader)``.

    The training loader drops the last partial batch, so every optimizer step
    sees the same token count -- which keeps the delta rule's ``Sigma_hat``
    normalization uniform across steps.
    """
    train_set = CIFAR10(root=data_root, train=True, download=True, transform=train_transform())
    test_set = CIFAR10(root=data_root, train=False, download=True, transform=test_transform())

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader
