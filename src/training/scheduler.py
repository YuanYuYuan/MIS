from torch.optim.lr_scheduler import _LRScheduler


class Scheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        warmup=0,
        reduce_factor=1.,
        threshold=0.95,
        patience=0,
        mode='min'
    ):

        # warmup
        self.use_warmup = warmup > 0
        if self.use_warmup:
            self.warmup = warmup

        # reduce learning rate on plateau
        self.use_reduce_lr = reduce_factor < 1 and patience > 0
        if self.use_reduce_lr:
            self.reduce_factor = reduce_factor
            self.patience = patience
            self.n_stagnation = 0
            self.reduced_lrs = None
            assert mode in ['min', 'max']
            self.mode = mode
            self.best = None

        self.stage = 'init'
        self.threshold = threshold
        super().__init__(optimizer)

    def get_lr(self):
        if self.use_warmup and self.last_epoch <= self.warmup:
            self.stage = 'warmup'
            return [
                lr * self.last_epoch / self.warmup
                for lr in self.base_lrs
            ]
        elif self.use_reduce_lr:
            self.stage = 'reducing'
            return self.reduced_lrs
        else:
            self.stage = 'constant'
            return self.base_lrs

    def check_improved(self, metric):
        if self.best is None:
            return True
        elif self.mode == 'min':
            return (metric * self.threshold) < self.best
        else:
            return metric > (self.best * self.threshold)

    def step(self, epoch=None, metric=None):
        if self.stage == 'init':
            if self.use_reduce_lr:
                self.reduced_lrs = self.base_lrs

        elif self.stage in ['constant', 'warmup']:
            pass

        elif self.stage == 'reducing':
            if not isinstance(metric, float):
                metric = float(metric)
            if self.check_improved(metric):
                self.best = metric
            else:
                self.n_stagnation += 1
                if self.n_stagnation > self.patience:
                    self.reduced_lrs = [
                        lr * self.reduce_factor
                        for lr in self.reduced_lrs
                    ]
                    self.n_stagnation = 0
        else:
            raise ValueError(self.stage)
        super().step(epoch)
