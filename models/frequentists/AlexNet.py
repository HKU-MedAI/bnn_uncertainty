import torch.nn as nn
from torch.nn import functional as F
from layers import (
    FlattenLayer
)


class AlexNet(nn.Module):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, activation_type='softplus'):
        super(AlexNet, self).__init__()

        self.num_classes = outputs

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.conv1 = nn.Conv2d(inputs, 64, 11, stride=4, padding=2, bias=True)
        self.act1 = self.act()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(64, 192, 5, padding=2, bias=True)
        self.act2 = self.act()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(192, 384, 3, padding=1, bias=True)
        self.act3 = self.act()

        self.conv4 = nn.Conv2d(384, 256, 3, padding=1, bias=True)
        self.act4 = self.act()

        self.conv5 = nn.Conv2d(256, 256, 3, padding=1, bias=True)
        self.act5 = self.act()
        self.pool3 = nn.MaxPool2d(kernel_size=1, stride=2)

        self.flatten = FlattenLayer(1 * 1 * 256)
        self.fc = nn.Linear(1 * 1 * 256, outputs, bias=True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.act3(x)

        x = self.conv4(x)
        x = self.act4(x)

        x = self.conv5(x)
        x = self.act5(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.fc(x)

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

        x = self.conv3(x)
        x = self.act3(x)

        x = F.dropout(x, p)

        x = self.conv4(x)
        x = self.act4(x)

        x = F.dropout(x, p)

        x = self.conv5(x)
        x = self.act5(x)
        x = self.pool3(x)

        x = F.dropout(x, p)

        x = self.flatten(x)
        x = self.fc(x)

        return x