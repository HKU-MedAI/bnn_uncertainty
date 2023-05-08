from pathlib import Path
from os.path import join
from PIL import Image

import torch
import torchvision
from torchvision import transforms

from data.core import split_dataset
from utils.config import Config

import pickle

from torch.utils.data import TensorDataset, Dataset


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


def diabetes(split='train'):
    if split not in ['train', 'val', 'test', 'all']:
        raise ValueError(f'Invalid split: {split}')
    cfg = Config()
    # train = (split == 'train') or (split == 'val')
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomHorizontalFlip(),
            norm,
            torchvision.transforms.Resize((64, 64))
        ])
    train = DiabetesDataset("../../data/diabetes/train_ID", transform)
    val = DiabetesDataset("../../data/diabetes/train_ID", transform)
    temp = DiabetesDataset("../../data/diabetes/test_ID", transform)
    test = DiabetesDataset("../../data/diabetes/test_OOD", transform)
    test.data_paths += temp.data_paths
    if split != 'train':
        if split == 'val':
            return val
        elif split == 'test':
            return test
        return train, val, test
    return train

