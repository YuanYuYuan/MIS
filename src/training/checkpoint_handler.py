from .model_handler import ModelHandler
from glob import glob
import os

MODES = [
    'each',
    'best',
    'improved'
]

CRITERION = ['max', 'min']


class CheckpointHandler:

    def __init__(
        self,
        model_handler: ModelHandler,
        checkpoint_dir,
        mode='improved',
        criterion='max',
    ):

        assert mode in MODES, (mode, MODES)
        self.mode = mode
        if mode in ['improved', 'best']:
            assert criterion in CRITERION, (criterion, CRITERION)
            self.criterion = criterion
            self.best = None
        self.model_handler = model_handler
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

    def check_improved(self, metric):
        if self.best is None:
            return True
        elif self.criterion == 'min':
            return metric <= self.best
        else:
            return metric >= self.best

    def run(
        self,
        metric: float,
        epoch: int = int(),
        additional_info: dict = dict()
    ):

        assert metric is not None
        if self.mode == 'each':
            file_name = '%.5f.pt' % metric
            if epoch != int():
                file_name = '%03d_' % epoch + file_name

            self.model_handler.save(
                file_path=os.path.join(self.checkpoint_dir, file_name),
                additional_info=additional_info
            )

        elif self.check_improved(metric):
            self.best = metric
            file_name = '%.5f.pt' % metric
            if epoch != int():
                file_name = '%03d_' % epoch + file_name

            if self.mode == 'best':
                file_name = 'best_' + file_name
                for f in glob(os.path.join(
                    self.checkpoint_dir,
                    'best_*'
                )):
                    os.remove(f)
            self.model_handler.save(
                file_path=os.path.join(self.checkpoint_dir, file_name),
                additional_info=additional_info
            )
