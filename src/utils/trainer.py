import torch
from metrics import dice_score
from tqdm import tqdm
from .utils import get_tty_columns


'''
For training only and use GPU by default
'''


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        accu_fn='dice',
        load_checkpoint=None,
        logger=None,
    ):

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.accu_fn = dice_score if accu_fn == 'dice' else accu_fn
        self.model = model

        # TensorboardX logger
        self.logger = logger

        # load checkpoint
        if load_checkpoint is not None:
            print('===== Loading checkpoint %s =====' % load_checkpoint)

            checkpoint = torch.load(
                load_checkpoint,
                map_location=lambda storage, location: storage
            )

            # check validity between fresh model state and pretrained one
            model_state = self.model.state_dict()
            pretrained_weights = dict()
            for key, val in checkpoint['model_state_dict'].items():
                if key not in model_state:
                    print('Exclude %s since not in current model state' % key)
                else:
                    if val.size() != model_state[key].size():
                        print('Exclude %s due to size mismatch' % key)
                    else:
                        pretrained_weights[key] = val

            model_state.update(pretrained_weights)
            self.model.load_state_dict(model_state)
            self.init_epoch = checkpoint['epoch']
            self.global_step = checkpoint['step']
        else:
            self.init_epoch = 0
            self.global_step = 0

        # swicth between multi_gpu/single gpu modes
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model).cuda()
        else:
            self.model = self.model.cuda()


    def train(self, batch):
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

    def run(self, data_gen, epoch):

        running_loss = 0.0
        running_accu = 0.0

        n_steps = len(data_gen)

        progress_bar = tqdm(
            enumerate(data_gen),
            total=n_steps,
            ncols=get_tty_columns(),
            dynamic_ncols=True,
            desc='[Training] Loss: %.5f, Accu: %.5f'
            % (0.0, 0.0)
        )

        for step, batch in progress_bar:
            loss, accu = self.train(batch)
            progress_bar.set_description(
                '[Training]  Loss: %.5f, Accu: %.5f'
                % (loss.item(), accu.item())
            )

            running_loss += loss.item()
            running_accu += accu.item()

            if self.logger is not None:
                self.logger.add_scalar(
                    'metrics/step_loss',
                    loss.item(),
                    self.global_step+1
                )
                self.logger.add_scalar(
                    'metrics/step_accu',
                    accu.item(),
                    self.global_step+1
                )

            self.global_step += 1

        running_loss /= n_steps
        running_accu /= n_steps


        return running_loss, running_accu
