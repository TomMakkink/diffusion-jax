"""Module containing functions to create an image dataset and visualise the images."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import ConcatDataset
from torchvision import transforms

from diffusion.ddpm.constants import IMG_SIZE


def load_transformed_dataset() -> ConcatDataset[tuple[torch.Tensor, int]]:
    """Loads a torchvision dataset, performs transformations on it,
    creates a training set and a test set, and finally combines these
    into one object.

    Returns:
        `ConcatDataset` object where each element is a tuple consisting of
            - The transformed image.
            - The label.
    """
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data to be in [0, 1]
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scales data to be in [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.OxfordIIITPet(
        root=".", download=True, transform=data_transform
    )
    test = torchvision.datasets.OxfordIIITPet(
        root=".", download=True, transform=data_transform, split="test"
    )
    return torch.utils.data.ConcatDataset([train, test])


def show_tensor_image(image: Image) -> None:
    reverse_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            transforms.Lambda(lambda t: t * 255.0),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ]
    )

    # Pick the first image in the batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))
