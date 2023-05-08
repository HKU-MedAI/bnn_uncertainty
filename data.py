from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms

from PIL import Image

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """

        img = np.array(img)
        h, w, c = img.shape
        mask = np.ones((h, w, c), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        img = img * mask

        img = Image.fromarray(np.uint8(img))

        return img


def load_data(name, train, image_size=32, in_channel=1, transform="pos"):

    if in_channel == 3:
        norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    else:
        norm = transforms.Normalize((0.1307,), (0.3081,))

    if transform == "pos":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Grayscale(num_output_channels=in_channel),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomHorizontalFlip(),
                norm,
                torchvision.transforms.Resize((image_size, image_size))
            ])
    elif transform == "neg":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Grayscale(num_output_channels=in_channel),
                Cutout(n_holes=10, length=5),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(90),
                norm,
                torchvision.transforms.Resize((image_size, image_size))
            ])
    elif transform is None:
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Grayscale(num_output_channels=in_channel),
                torchvision.transforms.ToTensor(),
                norm,
                torchvision.transforms.Resize((image_size, image_size))
            ])

    if name == "MNIST":
        d = torchvision.datasets.MNIST
    elif name == "FashionMNIST":
        d = torchvision.datasets.FashionMNIST
    elif name == "CIFAR10":
        d = torchvision.datasets.CIFAR10
    elif name == "CIFAR100":
        d = torchvision.datasets.CIFAR100
    elif name == "Omniglot":
        d = torchvision.datasets.Omniglot
        return d(root='./data', background=train, download=True, transform=transform)
    elif name == "SVHN":
        train = "train" if train else "test"
        d = torchvision.datasets.SVHN
        return d(root='./data', split=train, download=True, transform=transform)
    elif "diabetes" in name:
        if "test_in" in name:
            img_dir = f"./data/diabetes/test_ID"
        elif "train" in name:
            img_dir = f"./data/diabetes/train_ID"
        elif "test_ood" in name:
            img_dir = f"./data/diabetes/test_OOD"
        else:
            raise ValueError
        return DiabetesDataset(img_dir, transform)
    else:
        raise NotImplementedError

    return d(root='./data', train=train, download=True, transform=transform)

class DiabetesDataset(Dataset):
    def __init__(self, img_dir: str, transform) -> None:
        self.data_paths = [p for p in Path(img_dir).iterdir()]
        self.transform = transform

    def __getitem__(self, idx: int):
        img_path = str(self.data_paths[idx])
        img = Image.open(img_path).convert('RGB')

        if "left" in img_path:
            label = 1
        elif "right" in img_path:
            label = 0
        else:
            raise ValueError

        img = self.transform(img)

        return img, label

    def __len__(self) -> int:
        return len(self.data_paths)
