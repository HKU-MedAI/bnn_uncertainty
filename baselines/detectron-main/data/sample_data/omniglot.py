from os.path import join

import torch
import torchvision
from torchvision import transforms

from data.core import split_dataset
from utils.config import Config

import pickle

from torch.utils.data import TensorDataset, Dataset


def omniglot(split='train'):
    if split not in ['train', 'val', 'test', 'all']:
        raise ValueError(f'Invalid split: {split}')
    # train = (split == 'train') or (split == 'val')
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomHorizontalFlip(),
            norm,
            torchvision.transforms.Resize(32)
        ])
    train, rest = [torchvision.datasets.Omniglot(
        root="./data",
        background=x,
        download=True,
        transform=transform
    ) for x in [True, False]]

    if split != 'train':
        val, test = split_dataset(rest, num_samples=int(len(rest) * 9 / 10), random_seed=0)
        if split == 'val':
            return val
        elif split == 'test':
            return test
        return train, val, test
    return train