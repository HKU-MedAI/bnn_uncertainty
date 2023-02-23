import math
from torchvision.models import resnet18

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import (
    BBBLinear,
    BBBConv2d
)


class BBBResNet(nn.Module):
    def __init__(self, outputs, inputs, image_size=256, priors=None, activation_type='softplus'):
        super(BBBResNet, self).__init__()

        self.num_classes = outputs
        self.priors = priors

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        model = resnet18(pretrained=True)

        model.conv1 = self.freq_to_bayes(model.conv1, "conv")
        model.layer1[0].conv1 = self.freq_to_bayes(model.layer1[0].conv1, "conv")
        model.layer2[0].conv1 = self.freq_to_bayes(model.layer2[0].conv1, "conv")
        model.fc = self.freq_to_bayes(model.fc, "linear")

        self.model = model

        self.out = BBBLinear(1000, outputs, priors=priors)

    def freq_to_bayes(self, ly, tp):
        """
        Turn frequentist layers into Bayesian
        :param ly:
        :param tp: type of the layer (conv or linear)
        :return:
        """
        if tp == "conv":
            bconv = BBBConv2d(ly.in_channels, ly.out_channels, ly.kernel_size, stride=ly.stride,
                              padding=ly.padding[0], bias=False, priors=self.priors)
            bconv.W_mu = ly.weight
            return bconv
        elif tp == "linear":
            blinear = BBBLinear(ly.in_features, ly.out_features, bias=False, priors=self.priors)
            blinear.W_mu = ly.weight
            return blinear
        else:
            raise NotImplementedError

    def forward(self, x):

        emb = self.model(x)
        out = self.out(emb)

        kl = 0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return out, kl, emb

    def inference(self, x):
        maps = []
        maps.append(x.cpu().numpy())

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        maps.append(x.cpu().numpy())

        x = self.model.layer1(x)
        maps.append(x.cpu().numpy())

        x = self.model.layer2(x)
        maps.append(x.cpu().numpy())

        return maps