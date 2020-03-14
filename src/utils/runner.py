import torch
from metrics import dice_score
from tqdm import tqdm
from .utils import get_tty_columns


class Runner:

    def __init__(
        self,
        model,
        loss_fn,
        optimizer=None,
        stage='train',
        accu_fn='dice',
        logger=None,
        step=0,
    ):
        # sanity check
        assert stage in ['train', 'valid']
        self.stage = stage
        if stage == 'train':
            assert optimizer is not None

        self.optimizer = optimizer
        self.accu_fn = dice_score if accu_fn == 'dice' else accu_fn
        self.loss_fn = loss_fn
        self.model = model
        self.logger = logger
        self.step = step

    def process_batch(self, batch):

        if self.stage == 'train':
            with torch.set_grad_enabled(True):
                self.model.train()
                self.optimizer.zero_grad()

                images = batch['image'].cuda()
                labels = batch['label'].cuda()
                outputs = self.model(images)

                # metrics
                loss = self.loss_fn(outputs, labels)
                accu = self.accu_fn(outputs, labels).mean()

                # back propagation
                loss.backward()
                self.optimizer.step()

            return loss, accu

        else:
            with torch.set_grad_enabled(False):
                self.model.eval()

                images = batch['image'].cuda()
                labels = batch['label'].cuda()
                outputs = self.model(images)

                # metrics
                loss = self.loss_fn(outputs, labels)
                accu = self.accu_fn(outputs, labels).mean()

            return loss, accu

    def run(self, data_gen):

        running_loss = 0.0
        running_accu = 0.0

        n_steps = len(data_gen)

        progress_bar = tqdm(
            enumerate(data_gen),
            total=n_steps,
            ncols=get_tty_columns(),
            dynamic_ncols=True,
            desc='[%s] Loss: %.5f, Accu: %.5f'
            % (self.stage, 0.0, 0.0)
        )

        for step, batch in progress_bar:
            loss, accu = self.process_batch(batch)
            progress_bar.set_description(
                '[%s]  Loss: %.5f, Accu: %.5f'
                % (self.stage, loss.item(), accu.item())
            )

            running_loss += loss.item()
            running_accu += accu.item()

            if self.logger is not None:
                self.logger.add_scalar(
                    '%s/metrics/step_loss' % self.stage,
                    loss.item(),
                    self.step+1
                )
                self.logger.add_scalar(
                    '%s/metrics/step_accu' % self.stage,
                    accu.item(),
                    self.step+1
                )
            self.step += 1

        running_loss /= n_steps
        running_accu /= n_steps

        return {
            'loss': running_loss,
            'accu': running_accu,
        }
