import json5
from .chain import Chain
import torch.nn as nn


class NewVNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        with open(config) as f:
            model = json5.load(f)
        self.encoder = Chain(**model['encoder'])
        self.decoder = Chain(**model['decoder'])

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x[0]
