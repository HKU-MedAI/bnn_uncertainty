import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, output_dim=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(500, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, output_dim),
        )

    def forward(self, x):
        conv_features = self.features(x)
        flatten = conv_features.view(conv_features.size(0), -1)
        fc = self.fc_layers(flatten)
        return fc


def cnn(output_dim, **kwargs):
    return CNN(output_dim, **kwargs)
