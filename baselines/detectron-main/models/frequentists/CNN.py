import torch.nn as nn
from torch.nn import functional as F
from layers import (
    FlattenLayer
)


class CNN(nn.Module):
    """
    Simple Neural Network having 3 Convolution
    and 3 FC layers with Bayesian layers.
    """

    def __init__(self, outputs, inputs, image_size=32, activation_type='softplus', get_sig=False):
        super(CNN, self).__init__()

        self.get_sig = get_sig
        self.num_classes = outputs

        if activation_type == 'softplus':
            self.act = nn.Softplus()
        elif activation_type == 'relu':
            self.act = nn.ReLU()
        else:
            raise ValueError("Only softplus or relu supported")

        self.conv1 = nn.Conv2d(inputs, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, outputs)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.flat = nn.Flatten()

        if get_sig:
            self.fc_sig = nn.Linear(50, outputs, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.conv2_drop(x)
        x = self.act(x)

        x = self.flat(x)
        x = self.fc1(x)
        x = self.act(x)
        out = self.fc2(x)

        if self.get_sig:
            out_sig = self.fc_sig(x)
            return out, out_sig

        return out

    def mc_dropout(self, x, p):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = F.dropout(x, p=p)
        x = self.act(x)

        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.conv2_drop(x)
        x = self.act(x)

        x = self.flat(x)
        x = self.fc1(x)
        x = F.dropout(x, p=p)
        x = self.act(x)
        x = self.fc2(x)

        return x