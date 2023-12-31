import math
import torch.nn as nn

from layers import (
    BBBConv2d,
    BBBLinear,
    FlattenLayer,
)


class BBBLeNet(nn.Module):
    '''The architecture of LeNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, priors, image_size=32, activation_type='softplus', de=False, p=84):
        """

        :param outputs:
        :param inputs:
        :param priors:
        :param image_size:
        :param activation_type:
        :param de: Whether apply deep ensembles or not
        """
        super(BBBLeNet, self).__init__()

        self.num_classes = outputs
        self.priors = priors

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.conv1 = BBBConv2d(inputs, 6, 5, padding=0, bias=True, priors=self.priors)
        self.act1 = self.act()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        out_size = (image_size - 5 + 1) // 2

        self.conv2 = BBBConv2d(6, 16, 5, padding=0, bias=True, priors=self.priors)
        self.act2 = self.act()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        out_size = (out_size - 5 + 1) // 2

        self.flatten = FlattenLayer(out_size * out_size * 16)
        self.fc1 = BBBLinear(out_size * out_size * 16, 120, bias=True, priors=self.priors)
        self.act3 = self.act()

        self.fc2 = BBBLinear(120, p, bias=True, priors=self.priors)
        self.act4 = self.act()

        self.fc3 = BBBLinear(p, outputs, bias=True, priors=self.priors)

        self.is_de = de
        if de:
            self.fc_sig = BBBLinear(84, outputs, bias=True, priors=self.priors)

    def forward(self, x, get_emb=False):

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

        out = self.fc3(x)

        if get_emb:
            return x

        if self.is_de:
            out_sig = self.fc_sig(x)
            return out, out_sig

        return out

    def kl_loss(self):
        # Compute KL divergences
        kl = 0.0
        for module in self.children():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return kl
