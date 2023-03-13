import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms


def load_data(name, train, image_size=32, in_channel=1):
    if in_channel == 3:
        norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    else:
        norm = transforms.Normalize((0.1307,), (0.3081,))
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(num_output_channels=in_channel),
            torchvision.transforms.ToTensor(),
            norm,
            torchvision.transforms.Resize(image_size)
        ])
    if name == "MNIST":
        d = torchvision.datasets.MNIST
    elif name == "FashionMNIST":
        d = torchvision.datasets.FashionMNIST
    elif name == "CIFAR10":
        d = torchvision.datasets.CIFAR10
    elif name == "CIFAR100":
        d = torchvision.datasets.CIFAR100
    elif name == "Omiglot":
        d = torchvision.datasets.Omniglot
        return d(root='./data', background=train, download=True, transform=transform)
    elif name == "SVHN":
        train = "train" if train else "test"
        d = torchvision.datasets.SVHN
        return d(root='./data', split=train, download=True, transform=transform)
    else:
        raise NotImplementedError

    return d(root='./data', train=train, download=True, transform=transform)
