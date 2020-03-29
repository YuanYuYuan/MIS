import torch
from tqdm import tqdm
from utils import get_tty_columns


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

    def process_batch(self, batch, training=True):

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

        # image = batch['image'].cuda()
        # label = batch['label'].cuda()
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

        for key in results:
            results[key] = results[key].detach().cpu()
        return results

    def run(self, data_gen, training=True, stage=None):
        if stage is None:
            stage = 'train' if training else 'valid'
        n_steps = len(data_gen)

        progress_bar = tqdm(
            enumerate(data_gen),
            total=n_steps,
            ncols=get_tty_columns(),
            dynamic_ncols=True,
            desc='[%s] Loss: %.5f, Accu: %.5f'
            % (stage, 0.0, 0.0)
        )

        if stage not in self.step:
            self.step[stage] = 1

        result_list = []
        for step, batch in progress_bar:

            self.step[stage] += 1
            if self.logger is not None:
                ratio = (batch['label'] > 0).float().mean().item()
                self.logger.add_scalar(
                    '%s/quality/ratio' % stage,
                    ratio,
                    self.step[stage]
                )
                if ratio == 0:
                    continue


            result = self.process_batch(batch, training=training)
            step_loss = result['loss'].item()
            step_accu = result['accu'].mean().item()

            progress_bar.set_description(
                '[%s] Loss: %.5f, Avg accu: %.5f'
                % (stage, step_loss, step_accu)
            )

            if self.logger is not None:
                self.logger.add_scalar(
                    '%s/step/loss' % stage,
                    step_loss,
                    self.step[stage]
                )
                self.logger.add_scalar(
                    '%s/step/accu' % stage,
                    step_accu,
                    self.step[stage]
                )

            if step_accu >= 0.:
                result_list.append(result)

        return result_list
