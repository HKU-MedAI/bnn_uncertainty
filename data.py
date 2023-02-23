import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms


def load_data(name, train, batch_size):
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(
             (0.1307,), (0.3081,)),
         ])
    if name == "MNIST":
        d = torchvision.datasets.MNIST
    elif name == "CIFAR10":
        d = torchvision.datasets.CIFAR10
    elif name == "CIFAR100":
        d = torchvision.datasets.CIFAR100
    elif name == "Omiglot":
        d = torchvision.datasets.Omniglot
    elif name == "FashionMNIST":
        d = torchvision.datasets.FashionMNIST
    elif name == "SVHN":
        d = torchvision.datasets.SVHN
    else:
        raise NotImplementedError

    return d(root='./data', train=train, download=True, transform=transform)
