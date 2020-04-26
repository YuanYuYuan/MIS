from tqdm import tqdm
from utils import get_tty_columns
from .learners import SegLearner, AdvSegLearner, DisLearner
import math
import numpy as np


class Runner:

    def __init__(self, learner: SegLearner, logger=None):
        self.learner = learner
        self.logger = logger
        self.step = dict()

    def run(
        self,
        data_gen,
        training=True,
        stage=None,
        min_ratio=0.,
        include_prediction=False,
    ):
        if stage is None:
            stage = 'train' if training else 'valid'
        n_steps = len(data_gen)

        progress_bar = tqdm(
            data_gen,
            total=n_steps,
            ncols=get_tty_columns(),
            dynamic_ncols=True,
            desc='[%s] Loss: %.5f, Accu: %.5f'
            % (stage, 0.0, 0.0)
        )

        if stage not in self.step:
            self.step[stage] = 1

        result_list = []
        for batch in progress_bar:

            self.step[stage] += 1
            if self.logger is not None:
                ratio = (batch['label'] > 0).float().mean().item()
                self.logger.add_scalar(
                    '%s/quality/ratio' % stage,
                    ratio,
                    self.step[stage]
                )
                if ratio < min_ratio:
                    continue

            data = {
                'image': batch['image'].cuda(),
                'label': batch['label'].cuda()
            }

            if training:
                results = self.learner.learn(data)
            else:
                results = self.learner.infer(
                    data,
                    include_prediction=include_prediction,
                    compute_match=(not include_prediction)
                )

            # detach all, move to CPU, and convert to numpy
            for key in results:
                results[key] = results[key].detach().cpu().numpy()

            if 'accu' in results:
                step_accu = np.nanmean(results['accu'])
            else:
                step_accu = math.nan

            progress_bar.set_description(
                '[%s] Loss: %.5f, Avg accu: %.5f'
                % (stage, results['loss'], step_accu)
            )

            if self.logger is not None:
                self.logger.add_scalar(
                    '%s/step/loss' % stage,
                    results['loss'],
                    self.step[stage]
                )
                self.logger.add_scalar(
                    '%s/step/accu' % stage,
                    -1 if math.isnan(step_accu) else step_accu,
                    self.step[stage]
                )

            # if include_prediction:
            #     # XXX: deprecated
            #     # output_threshold = 0.3
            #     # prediction = results.pop('prediction')
            #     # for i in range(1, prediction.shape[1]):
            #     #     prediction[:, i, ...] += \
            #     #         (prediction[:, i, ...] >= output_threshold).astype(np.float)
            #     # # prediction[:, 1:, ...] = (prediction[:, 1:, ...] >= output_threshold).astype(np.float)
            #     # prediction = np.argmax(prediction, 1)
            #     # prediction_list.append(prediction)

            #     prediction_list.append(results.pop('prediction'))

            result_list.append(results)

        return result_list


# TODO: simplify the implementation by class inheritance
class AdvRunner:

    def __init__(
        self,
        learners,
        start_adv=1,
        logger=None,
    ):
        assert 'seg' in learners
        assert isinstance(learners['seg'], AdvSegLearner)
        assert 'dis' in learners
        assert isinstance(learners['dis'], DisLearner)
        self.learners = learners
        self.logger = logger
        self.step = dict()
        self.start_adv = start_adv

    def run(
        self,
        data_gen,
        training=True,
        stage=None,
        unlabeled=False,  # FIXME
        train_dis=False,  # FIXME
        min_ratio=0.,
        include_prediction=False,
    ):
        if stage is None:
            stage = 'train' if training else 'valid'
        n_steps = len(data_gen)

        progress_bar = tqdm(
            data_gen,
            total=n_steps,
            ncols=get_tty_columns(),
            dynamic_ncols=True,
            desc='[%s] Loss: %.5f, Accu: %.5f'
            % (stage, 0.0, 0.0)
        )

        if stage not in self.step:
            self.step[stage] = 1

        result_list = []
        for batch in progress_bar:

            self.step[stage] += 1
            if self.logger is not None:
                ratio = (batch['label'] > 0).float().mean().item()
                self.logger.add_scalar(
                    '%s/quality/ratio' % stage,
                    ratio,
                    self.step[stage]
                )
                if ratio < min_ratio:
                    continue

            data = {
                'image': batch['image'].cuda(),
                'label': batch['label'].cuda()
            }

            # FIXME
            if train_dis:
                results = self.learners['seg'].infer(data)
                results.update(self.learners['dis'].learn(data))
            else:
                if training:
                    results = self.learners['seg'].learn(data)
                    if not unlabeled:  # FIXME
                        if self.step[stage] >= self.start_adv:
                            results.update(self.learners['dis'].learn(data))
                else:
                    results = self.learners['seg'].infer(
                        data,
                        include_prediction=include_prediction,
                        compute_match=(not include_prediction)
                    )

            # detach all, move to CPU, and convert to numpy
            for key in results:
                results[key] = results[key].detach().cpu().numpy()

            if 'accu' in results:
                step_accu = np.nanmean(results['accu'])
            else:
                step_accu = math.nan

            progress_bar.set_description(
                '[%s] Loss: %.5f, Avg accu: %.5f'
                % (stage, results['loss'], step_accu)
            )

            if self.logger is not None:
                self.logger.add_scalar(
                    '%s/step/loss' % stage,
                    results['loss'],
                    self.step[stage]
                )
                self.logger.add_scalar(
                    '%s/step/accu' % stage,
                    -1 if math.isnan(step_accu) else step_accu,
                    self.step[stage]
                )
            result_list.append(results)

        return result_list
