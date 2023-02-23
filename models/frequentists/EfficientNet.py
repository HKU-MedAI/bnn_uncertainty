import torch.nn as nn

from efficientnet_pytorch import EfficientNet

class EfficientNetB4(nn.Module):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, outputs, inputs):
        super(EfficientNetB4, self).__init__()

        self.encoder = EfficientNet.from_pretrained('efficientnet-b4', num_classes=256, in_channels=inputs)

        self.out = nn.Linear(256, outputs)
