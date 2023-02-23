import math
from torchvision.models import resnet18

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, outputs, inputs, image_size=256, activation_type='softplus'):
        super(ResNet, self).__init__()

        self.num_classes = outputs

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.model = resnet18(pretrained=True)

        self.out = nn.Linear(1000, outputs)

    def forward(self, x):

        emb = self.model(x)
        out = self.out(emb)

        return out, emb

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