import math
from torchvision.models import resnet50

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, outputs, inputs, image_size=256, activation_type='softplus', get_sig=False):
        super(ResNet, self).__init__()

        self.get_sig = get_sig
        self.num_classes = outputs

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.model = resnet50(pretrained=True)

        self.out = nn.Linear(1000, outputs)

        if get_sig:
            self.out_sig = nn.Linear(1000, outputs)

    def forward(self, x):

        emb = self.model(x)
        out = self.out(emb)

        if self.get_sig:
            out_sig = self.out_sig(emb)
            return out, out_sig

        return out

    def mc_dropout(self, x, p):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = F.dropout(x, p=p)

        x = self.model.layer1(x)
        x = F.dropout(x, p=p)

        x = self.model.layer2(x)
        x = F.dropout(x, p=p)

        x = self.model.layer3(x)
        x = F.dropout(x, p=p)

        x = self.model.layer4(x)
        x = F.dropout(x, p=p)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)

        x = self.out(x)

        return x