import json5
from .chain import Chain
import torch.nn as nn


# class NewVNet(nn.Module):

#     def __init__(self, config):
#         super().__init__()
#         with open(config) as f:
#             model = json5.load(f)
#         self.encoder = Chain(**model['encoder'])
#         self.decoder = Chain(**model['decoder'])

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x


class NewVNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        with open(config) as f:
            model = json5.load(f)
        self.cfg = {
            'inps': [0],
            'outs': [5],
            'flow': {
                'encoder': {'inps': [0], 'outs': [1, 2, 3, 4]},
                'decoder': {'inps': [1, 2, 3, 4], 'outs': [5]}
            }
        }
        a: dict = {
            name: Chain(**model[name])
            for name in ['encoder', 'decoder']
        }
        self.sub_modules = nn.ModuleDict(a)

    def forward(self, x):
        state = [None] * 6
        if not isinstance(x, list):
            x = [x]
        assert len(self.cfg['inps']) == len(x)

        for idx in self.cfg['inps']:
            state[idx] = x[idx]

        for module, flow in self.cfg['flow'].items():
            tmp = self.sub_modules[module]([state[i] for i in flow['inps']])
            for i, j in enumerate(flow['outs']):
                state[j] = tmp[i]

        for i in range(6):
            print(state[i].shape)
        return [state[idx] for idx in self.cfg['outs']][0]
