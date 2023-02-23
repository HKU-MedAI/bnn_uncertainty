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

    def __init__(self, outputs, inputs, image_size=32, n_blocks=3, activation_type='softplus'):
        super(CNN, self).__init__()

        self.num_classes = outputs

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.n_blocks = n_blocks

        convs = [
                nn.Conv2d(inputs, 32, 5, padding=2, bias=True),
                nn.Conv2d(32, 64, 5, padding=2, bias=True),
                nn.Conv2d(64, 128, 5, padding=1, bias=True),
                nn.Conv2d(128, 128, 2, padding=1, bias=True)
        ]

        pools = [
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]

        self.conv_block = nn.Sequential()

        out_size = image_size
        out_channels = 0

        for l in range(self.n_blocks):
            self.conv_block.add_module(f"conv{l}", convs[l])
            self.conv_block.add_module(f"act{l}", self.act())
            self.conv_block.add_module(f"pool{l}", pools[l])
            out_size = (out_size - 5 + 2 * convs[l].padding[0]) // 1 + 1
            out_size = (out_size - 3) // 2 + 1
            out_channels = convs[l].out_channels

        self.dense_block = nn.Sequential(
                FlattenLayer(out_size * out_size * out_channels),
                nn.Linear(out_size * out_size * out_channels, 1000, bias=True),
                self.act(),
                nn.Linear(1000, 1000, bias=True),
                self.act(),
                nn.Linear(1000, outputs, bias=True)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.dense_block(x)

        return x

    def mc_dropout(self, x, p):
        for l in range(self.n_blocks):

            x = self.conv_block[3 * l](x)
            x = self.conv_block[3 * l + 1](x)
            x = self.conv_block[3 * l + 2](x)
            x = F.dropout(x, p=p)

        x = self.dense_block(x)

        return x