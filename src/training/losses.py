import torch
import metrics
from utils import load_config


class Loss:

    def __init__(self, config):

        # setup config
        if isinstance(config, str):
            self.config = load_config(config)
        else:
            assert isinstance(config, dict)
            self.config = config

        # setup loss
        assert 'name' in self.config, self.config
        self.loss_name = self.config.pop('name')
        if hasattr(metrics, self.loss_name):
            self.loss = getattr(metrics, self.loss_name)
        elif hasattr(torch.nn.functional, self.loss_name):
            self.loss = getattr(torch.nn.functional, self.loss_name)
        else:
            raise ValueError

        # weighted loss
        if 'weight' in config:
            if not isinstance(config['weight'], list):
                self.weight = [config['weight']]
            else:
                self.weight = config['weight']
            self.weight = torch.tensor(self.weight).float()
            self.weight /= torch.sum(self.weight)
            if torch.cuda.device_count() > 0:
                self.weight = self.weight.cuda()
        else:
            self.weight = None

    def __call__(self, predi, label):
        if self.weight is not None:
            return self.loss(predi, label, self.weight)
        else:
            return self.loss(predi, label)


class MixedLoss:

    def __init__(self, config):

        # setup config
        if isinstance(config, str):
            self.config = load_config(config)
        else:
            assert isinstance(config, dict)
            self.config = config

        # construct losses
        self.losses = [
            Loss(loss_cfg)
            for loss_cfg in self.config['losses']
        ]

        # decrease coefficeints of losses over iterations
        if 'coef' in self.config:
            if 'iter_decay' in self.config:
                self.iter_decay = self.config['iter_decay']
                self.coef = self.config['coef']
                assert len(self.coef) == len(self.iter_decay)
                self.iter = 0
            else:
                total = sum(self.config['coef'])
                self.coef = [
                    c / total
                    for c in self.config['coef']
                ]
                self.iter_decay = None
                self.iter = None
        else:
            self.coef = None

    def __call__(self, predi, label):
        if self.coef:
            if self.iter_decay:
                factor = []
                for c, d in zip(self.coef, self.iter_decay):
                    if c > 0 and d > 0:
                        factor.append(max(c * (1 - self.iter/d), 0.0))
                    else:
                        factor.append(c)
                total = sum(factor)
                factor = [f/total for f in factor]
                self.iter += 1

            else:
                factor = self.coef

            return sum([
                f * loss(predi, label)
                for (f, loss) in zip(factor, self.losses)
            ])
        else:
            return sum([loss(predi, label)for loss in self.losses])
