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
        step=0
    ):
        self.model = model
        self.meter = meter
        self.optimizer = optimizer
        self.logger = logger
        self.step = step

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

        image = batch['image'].cuda()
        label = batch['label'].cuda()
        if training:
            with torch.set_grad_enabled(True):
                self.model.train()
                self.optimizer.zero_grad()

                outputs = self.model(image)

                if outputs[0].shape[2:] != label.shape[1:]:
                    outputs[0] = outputs[0][
                        crop_range(
                            outputs[0].shape[2:],
                            label.shape[1:]
                        )
                    ]

                loss, accu = self.meter(outputs + [label, image])

                # back propagation
                loss.backward()
                self.optimizer.step()

            return loss, accu

        else:
            with torch.set_grad_enabled(False):
                self.model.eval()

                outputs = self.model(image)

                if outputs[0].shape[2:] != label.shape[1:]:
                    for o, l in zip(outputs[0].shape[2:], label.shape[1:]):
                        assert o >= l
                    crop_range = (slice(None), slice(None))
                    crop_range += tuple(
                        slice((o-l) // 2, l + (o-l) // 2)
                        for o, l in zip(outputs[0].shape[2:], label.shape[1:])
                    )
                    outputs[0] = outputs[0][crop_range]

                loss, accu = self.meter(outputs + [label, image])

            return loss, accu

    def run(self, data_gen, training=True):
        stage = 'train' if training else 'valid'
        running_loss = running_accu = 0.0
        n_steps = len(data_gen)

        progress_bar = tqdm(
            enumerate(data_gen),
            total=n_steps,
            ncols=get_tty_columns(),
            dynamic_ncols=True,
            desc='[%s] Loss: %.5f, Accu: %.5f'
            % (stage, 0.0, 0.0)
        )

        for step, batch in progress_bar:
            loss, accu = self.process_batch(batch, training=training)

            # TODO: multiple classes
            accu = accu.mean()

            progress_bar.set_description(
                '[%s] Loss: %.5f, Avg accu: %.5f'
                % (stage, loss.item(), accu.item())
            )

            running_loss += loss.item()
            running_accu += accu.item()

            if self.logger is not None:
                self.logger.add_scalar(
                    '%s/metrics/step_loss' % stage,
                    loss.item(),
                    self.step+1
                )
                self.logger.add_scalar(
                    '%s/metrics/step_accu' % stage,
                    accu.item(),
                    self.step+1
                )
            if training:
                self.step += 1

        running_loss /= n_steps
        running_accu /= n_steps

        return {
            'loss': running_loss,
            'accu': running_accu,
        }
