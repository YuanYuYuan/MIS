import torch
from tqdm import tqdm
from utils import get_tty_columns
from metrics import match_up
import math
import numpy as np


class Runner:

    def __init__(
        self,
        model,
        meter,
        optimizer=None,
        logger=None,
    ):
        self.model = model
        self.meter = meter
        self.optimizer = optimizer
        self.logger = logger
        self.step = dict()

    def process_batch(
        self,
        batch,
        training=True,
        include_prediction=False,
        compute_match=False,
    ):

        def crop_range(prediction_shape, label_shape):
            assert len(prediction_shape) == 3
            assert len(label_shape) == 3
            for p, l in zip(prediction_shape, label_shape):
                assert p >= l
            crop_range = (slice(None), slice(None))
            crop_range += tuple(
                slice((p-l) // 2, l + (p-l) // 2)
                for p, l in zip(prediction_shape, label_shape)
            )
            return crop_range

        data = {
            'image': batch['image'].cuda(),
            'label': batch['label'].cuda()
        }

        if training:
            with torch.set_grad_enabled(True):
                self.model.train()
                self.optimizer.zero_grad()

                outputs = self.model(data)

                # crop if size mismatched
                if outputs['prediction'].shape[2:] != data['label'].shape[1:]:
                    outputs['prediction'] = outputs['prediction'][
                        crop_range(
                            outputs['prediction'].shape[2:],
                            data['label'].shape[1:]
                        )
                    ]

                data.update(outputs)
                results = self.meter(data)

                # back propagation
                results['loss'].backward()
                self.optimizer.step()

        else:
            with torch.set_grad_enabled(False):
                self.model.eval()

                outputs = self.model(data)

                # crop if size mismatched
                if outputs['prediction'].shape[2:] != data['label'].shape[1:]:
                    outputs['prediction'] = outputs['prediction'][
                        crop_range(
                            outputs['prediction'].shape[2:],
                            data['label'].shape[1:]
                        )
                    ]

                data.update(outputs)
                results = self.meter(data)

        if include_prediction:
            probas = torch.nn.functional.softmax(outputs['prediction'], dim=1)
            results.update({'prediction': probas})

        if compute_match:
            match, total = match_up(
                outputs['prediction'],
                data['label'],
                needs_softmax=True,
                batch_wise=True,
                threshold=-1,
            )
            results.update({'match': match, 'total': total})

        # detach all, move to CPU, and convert to numpy
        for key in results:
            results[key] = results[key].detach().cpu().numpy()

        return results

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

        if training:
            include_prediction = False
            compute_match = False
        else:
            include_prediction = include_prediction
            compute_match = not include_prediction

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

            result = self.process_batch(
                batch,
                training=training,
                include_prediction=include_prediction,
                compute_match=compute_match,
            )

            step_accu = np.nanmean(result['accu'])
            progress_bar.set_description(
                '[%s] Loss: %.5f, Avg accu: %.5f'
                % (stage, result['loss'], step_accu)
            )

            if self.logger is not None:
                self.logger.add_scalar(
                    '%s/step/loss' % stage,
                    result['loss'],
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
            #     # prediction = result.pop('prediction')
            #     # for i in range(1, prediction.shape[1]):
            #     #     prediction[:, i, ...] += \
            #     #         (prediction[:, i, ...] >= output_threshold).astype(np.float)
            #     # # prediction[:, 1:, ...] = (prediction[:, 1:, ...] >= output_threshold).astype(np.float)
            #     # prediction = np.argmax(prediction, 1)
            #     # prediction_list.append(prediction)

            #     prediction_list.append(result.pop('prediction'))

            result_list.append(result)

        return result_list
