import json5
from .chain import Chain
import torch.nn as nn


class NewVNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        with open(config) as f:
            model = json5.load(f)
        self.op = nn.Sequential(
            Chain(**model['encoder']),
            Chain(**model['decoder']),
        )

    def forward(self, x):
        return self.op(x)
