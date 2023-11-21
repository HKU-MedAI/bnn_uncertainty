import math
import torch.nn as nn
from torch.nn import functional as F


class MultipleLinear(nn.Module):
    """
    Simple Neural Network having 3 Convolution
    and 3 FC layers with Bayesian layers.
    """

    def __init__(self, outputs, inputs, n_blocks=3, get_sig=False, activation_type='softplus'):
        super(MultipleLinear, self).__init__()

        self.num_classes = outputs

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.n_blocks = n_blocks

        linears = [
                nn.Linear(inputs, 32),
                nn.Linear(32, 32),
                nn.Linear(32, 128),
                nn.Linear(128, 128)
        ]

        out_channel = inputs

        self.dense_block = nn.Sequential()

        for l in range(self.n_blocks):
            self.dense_block.add_module(f"fc{l}", linears[l])
            self.dense_block.add_module(f"act{l}", self.act())
            out_channel = linears[l].out_features

        self.fc_out = nn.Linear(out_channel, outputs, bias=True)

        self.get_sig = get_sig
        if get_sig:
            self.out_sig = nn.Linear(out_channel, outputs, bias=True)

    def forward(self, x):
        emb = self.dense_block(x)

        out = self.fc_out(emb)
        if self.get_sig:
            out_sig = self.out_sig(emb)
            return out, out_sig

        return out

    def mc_dropout(self, x, p=0.2):
        for l in range(self.n_blocks):
            x = self.dense_block[2 * l](x)
            x = self.dense_block[2 * l + 1](x)
            x = F.dropout(x, p)
        x = self.fc_out(x)
        return x
