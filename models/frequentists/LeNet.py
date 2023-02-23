import torch.nn as nn
from torch.nn import functional as F

from layers import (
    FlattenLayer
)


class LeNet(nn.Module):
    '''The architecture of LeNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, image_size=32, activation_type='softplus'):
        super(LeNet, self).__init__()

        self.num_classes = outputs

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.conv1 = nn.Conv2d(inputs, 6, 5, padding=0, bias=True)
        self.act1 = self.act()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        out_size = (image_size - 5 + 1) // 2

        self.conv2 = nn.Conv2d(6, 16, 5, padding=0, bias=True)
        self.act2 = self.act()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        out_size = (out_size - 5 + 1) // 2

        self.flatten = FlattenLayer(out_size * out_size * 16)
        self.fc1 = nn.Linear(out_size * out_size * 16, 120, bias=True)
        self.act3 = self.act()

        self.fc2 = nn.Linear(120, 84, bias=True)
        self.act4 = self.act()

        self.fc3 = nn.Linear(84, outputs, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act3(x)

        x = self.fc2(x)
        x = self.act4(x)

        x = self.fc3(x)

        return x

    def mc_dropout(self, x, p):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = F.dropout(x, p)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = F.dropout(x, p)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act3(x)

        x = F.dropout(x, p)

        x = self.fc2(x)
        x = self.act4(x)

        x = F.dropout(x, p)

        x = self.fc3(x)

        return x