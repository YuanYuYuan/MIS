class EarlyStopper:

    def __init__(
        self,
        patience,
        threshold=0.95,
        mode='min',
    ):
        self.patience = patience
        self.n_stagnation = 0
        assert mode in ['min', 'max']
        self.mode = mode
        self.best = None
        self.threshold = threshold

    def check_improved(self, metric):
        if self.best is None:
            return True
        elif self.mode == 'min':
            return (metric * self.threshold) <= self.best
        else:
            return metric >= (self.best * self.threshold)

    def check(self, metric):
        early_stop_or_not = False

        improved = self.check_improved(metric)
        if improved:
            self.best = metric
            self.n_stagnation = 0
        else:
            self.n_stagnation += 1
            if self.n_stagnation > self.patience:
                early_stop_or_not = True

        return early_stop_or_not, improved
