import json5
import torch
from .flow import Flow
import metrics


class Metric:

    def __init__(self, name, **kwargs):
        if hasattr(metrics, name):
            self.metric = getattr(metrics, name)
        elif hasattr(torch.nn.functional, name):
            self.metric = getattr(torch.nn.functional, name)
        elif name == 'sum':
            self.metric = 'sum'
        else:
            raise ValueError
        self.kwargs = kwargs

    def __call__(self, inp: list):
        if self.metric == 'sum':
            if 'weight' in self.kwargs:
                return [sum(
                    i * w for i, w
                    in zip(inp, self.kwargs['weight'])
                )]
            else:
                return [sum(inp)]
        else:
            return [self.metric(*inp, **self.kwargs)]


class MetricFlow:

    def __init__(self, config):
        super().__init__()
        with open(config) as f:
            config = json5.load(f)
        self.flow = Flow(config, Metric)

    def __call__(self, x):
        return self.flow(x)
