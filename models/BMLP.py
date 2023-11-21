import math
import torch.nn as nn

from layers import (
    BBBLinear,
    FlattenLayer
)


class BBBMultipleLinear(nn.Module):
    """
    Simple Neural Network having 3 Convolution
    and 3 FC layers with Bayesian layers.
    """

    def __init__(self, outputs, inputs, priors, n_blocks=3, get_sig=False, activation_type='softplus'):
        super(BBBMultipleLinear, self).__init__()

        self.num_classes = outputs
        self.priors = priors

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.n_blocks = n_blocks

        linears = [
                BBBLinear(inputs, 32, priors=self.priors),
                BBBLinear(32, 32, priors=self.priors),
                BBBLinear(32, 128, priors=self.priors),
                BBBLinear(128, 128, priors=self.priors)
        ]

        out_channel = inputs

        self.dense_block = nn.Sequential()

        for l in range(self.n_blocks):
            self.dense_block.add_module(f"fc{l}", linears[l])
            self.dense_block.add_module(f"act{l}", self.act())
            out_channel = linears[l].out_features

        self.fc_out = BBBLinear(out_channel, outputs, bias=True, priors=self.priors)
        self.get_sig= get_sig
        if get_sig:
            self.fc_sig = BBBLinear(out_channel, outputs, bias=True, priors=self.priors)

    def kl_loss(self):
        # Compute KL divergences
        kl = 0.0
        for module in self.children():
            for cm in module.children():
                if hasattr(cm, 'kl_loss'):
                    kl = kl + cm.kl_loss()

        return kl

    def forward(self, x, get_emb=False):
        emb = self.dense_block(x)
        if get_emb:
            return emb

        out = self.fc_out(emb)
        if self.get_sig:
            out_sig = self.fc_sig(emb)
            return out, out_sig

        return out

