from torch import nn

class FlattenLayer(nn.Module):

    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.reshape(-1, self.num_features)


class ReverseFlattenLayer(nn.Module):

    def __init__(self, width, height, channel):
        super(ReverseFlattenLayer, self).__init__()
        self.width = width
        self.height = height
        self.channel = channel

    def forward(self, x):
        return x.view(x.size(0), int(self.channel / (self.width * self.height)), self.width, self.height)
