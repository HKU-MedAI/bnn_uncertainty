import torch
import torch.nn as nn
from torch.nn import functional as F
from layers import (
    BBBConv2d,
    BBBLinear,
    FlattenLayer
)


class BBBAlexNet(nn.Module):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, priors, image_size=32, activation_type='softplus'):
        super(BBBAlexNet, self).__init__()

        self.num_classes = outputs
        self.priors = priors

        if activation_type == 'softplus':
            self.act = nn.Softplus()
        elif activation_type == 'relu':
            self.act = nn.ReLU()
        else:
            raise ValueError("Only softplus or relu supported")

        self.convs = nn.ModuleList(
            (
                BBBConv2d(inputs, 64, 11, stride=4, padding=2, bias=True, priors=self.priors),
                BBBConv2d(64, 192, 5, padding=2, bias=True, priors=self.priors),
                BBBConv2d(192, 384, 3, padding=1, bias=True, priors=self.priors),
                BBBConv2d(384, 256, 3, padding=1, bias=True, priors=self.priors),
                BBBConv2d(256, 256, 3, padding=1, bias=True, priors=self.priors)
            )
        )

        self.pools = nn.ModuleList(
            (
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=1, stride=2)
            )
        )

        output_size_1 = ((image_size - 11 + 2 * 2) // 4 + 1) // 2
        output_size_2 = ((output_size_1 - 5 + 2 * 2) // 1 + 1) // 2
        output_size_3 = ((output_size_2 - 3 + 2 * 2) // 1 + 1) // 2

        self.flattens = nn.ModuleList(
            (
                FlattenLayer(output_size_1 * output_size_1 * 64),
                FlattenLayer(output_size_2 * output_size_2 * 192),
                FlattenLayer(output_size_3 * output_size_3 * 256)
            )
        )

        self.classifier = BBBLinear(output_size_3 * output_size_3 * 256, outputs, bias=True, priors=self.priors)

    def forward(self, x):

        x = self.convs[0](x)
        x = self.act(x)
        x = self.pools[0](x)

        x = self.convs[1](x)
        x = self.act(x)
        x = self.pools[1](x)

        x = self.convs[2](x)
        x = self.act(x)

        x = self.convs[3](x)
        x = self.act(x)

        x = self.convs[4](x)
        x = self.act(x)
        x = self.pools[2](x)

        x = self.flattens[2](x)

        x = self.classifier(x)

        return x

    def kl_loss(self):
        # Compute KL divergences
        kl = 0.0
        for module in self.children():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return kl

    def inference(self, x):
        """

        :param x: Data
        :return:
        """
        maps = []
        maps.append(x.cpu().numpy())
        x = self.convs[0](x)
        x = self.act(x)
        x = self.pools[0](x)
        maps.append(x.cpu().numpy())

        x = self.convs[1](x)
        x = self.act(x)
        maps.append(x.cpu().numpy())
        x = self.pools[1](x)

        x = self.convs[2](x)
        x = self.act(x)
        maps.append(x.cpu().numpy())

        x = self.convs[3](x)
        x = self.act(x)

        x = self.convs[4](x)
        x = self.act(x)
        x = self.pools[2](x)

        return maps
