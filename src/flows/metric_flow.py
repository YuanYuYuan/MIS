import json5
import torch
from .flow import Flow
import metrics
import inspect


# TODO: Write sum metric into metrics by a class
class Metric:

    def __init__(self, name, **kwargs):
        self.need_kwargs = True
        if hasattr(metrics, name):
            metric = getattr(metrics, name)
            if inspect.isclass(metric):
                self.need_kwargs = False
                self.metric = metric(**kwargs)
            else:
                self.metric = getattr(metrics, name)
        elif hasattr(torch.nn.functional, name):
            self.metric = getattr(torch.nn.functional, name)
        elif name == 'sum':
            self.metric = 'sum'
        else:
            raise ValueError

        if self.need_kwargs:
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
            if self.need_kwargs:
                return [self.metric(*inp, **self.kwargs)]
            else:
                return [self.metric(*inp)]


class MetricFlow:

    def __init__(self, config):
        super().__init__()
        with open(config) as f:
            config = json5.load(f)
        self.flow = Flow(config, Metric)

    def __call__(self, x):
        return self.flow(x)
