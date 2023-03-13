import math
import torch.nn as nn

from layers import (
    BBBConv2d,
    BBBLinear,
    FlattenLayer
)


class BBB3Conv3FC(nn.Module):
    """
    Simple Neural Network having 3 Convolution
    and 3 FC layers with Bayesian layers.
    """

    def __init__(self, outputs, inputs, priors, image_size=32, activation_type='softplus'):
        super(BBB3Conv3FC, self).__init__()

        self.num_classes = outputs

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.conv1 = BBBConv2d(inputs, 10, kernel_size=5, priors=priors)
        self.conv2 = BBBConv2d(10, 20, kernel_size=5, priors=priors)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = BBBLinear(320, 50, priors=priors)
        self.fc2 = BBBLinear(50, outputs, priors=priors)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.flat = nn.Flatten()

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
        x = self.fc2(x)

        return x

    def kl_loss(self):
        # Compute KL divergences
        kl = 0.0
        for module in self.children():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return kl
