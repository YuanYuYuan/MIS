import json5
import torch.nn as nn
from .flow import Flow
from models.chain import Chain


class ModuleFlow(nn.Module):

    def __init__(self, config):
        super().__init__()
        with open(config) as f:
            config = json5.load(f)
        self.flow = Flow(config, Chain)
        self.sub_modules = nn.ModuleDict(self.flow.nodes)

    def forward(self, x, **kwargs):
        return self.flow(x, nodes=self.sub_modules, **kwargs)
