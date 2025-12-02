from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_mnist_dataloaders(batch_size_images: int, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train, validation, and test dataloaders for MNIST.

    The training set is split into train/validation using a fixed seed for reproducibility.
    """
    transform = transforms.ToTensor()

    train_val_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    val_fraction = 0.1
    val_size = int(len(train_val_dataset) * val_fraction)
    train_size = len(train_val_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_images, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_images, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_images, shuffle=False)

    return train_loader, val_loader, test_loader
