from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


class _IndexedMNIST(datasets.MNIST):
    """MNIST dataset that returns (image, label, index)."""

    def __getitem__(self, index):  # type: ignore[override]
        image, label = super().__getitem__(index)
        return image, label, index


def get_mnist_dataloaders(batch_size_images: int, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train, validation, and test dataloaders for MNIST with dataset indices."""

    transform = transforms.ToTensor()

    train_val_dataset: Dataset = _IndexedMNIST(root="data", train=True, download=True, transform=transform)
    test_dataset: Dataset = _IndexedMNIST(root="data", train=False, download=True, transform=transform)

    val_fraction = 0.1
    val_size = int(len(train_val_dataset) * val_fraction)
    train_size = len(train_val_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_images, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_images, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_images, shuffle=False)

    return train_loader, val_loader, test_loader
