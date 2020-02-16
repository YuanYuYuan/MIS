import torch
from utils import load_config


class Optimizer:

    def __init__(self, config):

        if isinstance(config, str):
            self.config = load_config(config)
        else:
            assert isinstance(config, dict)
            self.config = config

        assert 'name' in self.config, self.config
        self.op = self.config.pop('name')
        if self.op not in ['SSO', 'curv']:
            assert hasattr(torch.optim, self.op), self.op

    def __call__(self, model):
        if self.op == 'SSO':
            from torchsso.optim import SecondOrderOptimizer
            return SecondOrderOptimizer(
                model,
                **self.config['optim_args'],
                curv_kwargs=self.config['curv_args']
            )
        elif self.op == 'curv':
            from torchcurv.optim import SecondOrderOptimizer
            return SecondOrderOptimizer(
                model,
                **self.config['optim_args'],
                **self.config['curv_args']
            )
        else:
            return getattr(torch.optim, self.op)(
                model.parameters(),
                **self.config
            )
