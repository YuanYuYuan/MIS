#!/usr/bin/env python3
from tensorboardX import SummaryWriter
import argparse
import time
import os
from tqdm import tqdm
from utils import get_tty_columns, epoch_info, EarlyStopper
import torch
import yaml
from training.optimizers import Optimizer
from training.scheduler import CosineAnnealingWarmUpRestarts as Scheduler
from MIDP import DataLoader, DataGenerator
from flows import MetricFlow, ModuleFlow


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
        self.zeros = None


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

        if training:
            if self.zeros is None:
                self.zeros = torch.zeros((image.shape[0],) + image.shape[2:]).long().cuda()
            with torch.set_grad_enabled(True):
                self.model.train()
                self.optimizer.zero_grad()

                outputs = self.model(image)
                loss, accu = self.meter(outputs + [self.zeros, image])

                # back propagation
                loss.backward()
                self.optimizer.step()

            return loss, accu

        else:
            label = batch['label'].cuda()
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


class ModelHandler:

    def __init__(
        self,
        model_config,
        checkpoint=None,
    ):
        self.model = ModuleFlow(model_config)

        # load checkpoint
        if checkpoint is None:
            self.checkpoint = None
        else:
            print('===== Loading checkpoint %s =====' % checkpoint)
            self.checkpoint = torch.load(
                checkpoint,
                map_location=lambda storage, location: storage
            )

            # check validity between fresh model state and pretrained one
            model_state = self.model.state_dict()
            pretrained_weights = dict()
            for key, val in self.checkpoint['model_state_dict'].items():
                if key not in model_state:
                    print('Exclude %s since not in current model state' % key)
                else:
                    if val.size() != model_state[key].size():
                        print('Exclude %s due to size mismatch' % key)
                    else:
                        pretrained_weights[key] = val

            model_state.update(pretrained_weights)
            self.model.load_state_dict(model_state)

        # swicth between multi_gpu/single gpu modes
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model).cuda()
        else:
            self.model = self.model.cuda()

    def save(self, file_path, additional_info=dict()):
        # check if multiple gpu model
        if torch.cuda.device_count() > 1:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        content = {'model_state_dict': model_state_dict}
        if len(additional_info) >= 1:
            content.update(additional_info)
        torch.save(content, file_path)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    required=True,
    help='training config'
)
parser.add_argument(
    '--checkpoint',
    default=None,
    help='pretrained model checkpoint'
)
parser.add_argument(
    '--checkpoint-dir',
    default='_ckpts',
    help='saved model checkpoints'
)
parser.add_argument(
    '--log-dir',
    default='_logs',
    help='training logs'
)
parser.add_argument(
    '--test',
    default=False,
    action='store_true',
    help='Small data test',
)
parser.add_argument(
    '--validate-only',
    default=False,
    action='store_true',
    help='do validation only',
)
parser.add_argument(
    '--pause-ckpt',
    help='save model checkpoint if paused'
)
args = parser.parse_args()

# load config
with open(args.config) as f:
    config = yaml.safe_load(f)
generator_config = config['generator']
with open(config['data']) as f:
    data_config = yaml.safe_load(f)
data_list = data_config['list']
loader_config = data_config['loader']

# - data pipeline
data_gen = dict()
loader_name = loader_config.pop('name')
ROIs = None
for stage in ['train', 'valid']:
    data_loader = DataLoader(loader_name, **loader_config)
    if data_list[stage] is not None:
        data_loader.set_data_list(data_list[stage])
    data_gen[stage] = DataGenerator(data_loader, generator_config[stage])

    if ROIs is None:
        ROIs = data_loader.ROIs

# - GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpus'])

# - model
model_handler = ModelHandler(
    config['model'],
    checkpoint=args.checkpoint,
)

# - optimizer
optimizer = Optimizer(config['optimizer'])(model_handler.model)

# - scheduler
if 'scheduler' in config:
    scheduler = Scheduler(
        optimizer,
        T_0=config['scheduler']['T_0'],
        T_mult=config['scheduler']['T_mult'],
        eta_max=config['optimizer']['lr'],
        T_up=config['scheduler']['T_up'],
        gamma=0.5
    )
else:
    scheduler = None


if args.log_dir is not None:
    logger = SummaryWriter(args.log_dir)
else:
    logger = None

timer = time.time()
start = timer

runner = Runner(
    model=model_handler.model,
    meter=MetricFlow(config['meter']),
    optimizer=optimizer,
    logger=logger,
)

checkpoint_dir = args.checkpoint_dir
if checkpoint_dir:
    os.makedirs(checkpoint_dir, exist_ok=True)

# early stopper
if config['early_stopping_epochs'] > 1:
    early_stopper = EarlyStopper(config['early_stopping_epochs'])
else:
    early_stopper = None


# set proper initial epoch
if model_handler.checkpoint is not None:
    init_epoch = model_handler.checkpoint['epoch']
else:
    init_epoch = 1

# main running loop
terminated = False
for epoch in range(init_epoch, init_epoch + config['epochs']):

    if terminated:
        break

    # epoch start
    epoch_info(epoch - 1, init_epoch + config['epochs'] - 1)

    for stage in ['train', 'valid']:
        training = True if stage == 'train' else False

        if stage == 'valid' and epoch % config['validation_frequency'] != 0:
            break
        # run on an epoch
        try:
            results = runner.run(data_gen[stage], training=training)
        except KeyboardInterrupt:
            print('save temporary model into %s' % args.pause_ckpt)
            model_handler.save(
                args.pause_ckpt,
                additional_info={'epoch': epoch, 'step': runner.step}
            )
            terminated = True
            break

        # summarize the performance
        print(', '.join(
            '%s: %.5f' % (key, val)
            for key, val in results.items()
        ))

        # record the performance
        if logger is not None:
            for key, val in results.items():
                logger.add_scalar(
                    '%s/metrics/%s' % (stage, key),
                    val,
                    epoch
                )

        # check early stopping
        if stage == 'valid' and early_stopper is not None:
            early_stop, improved = early_stopper.check(results['accu'])

            if early_stop:
                print('Early stopped.')
                terminated = True
                break

            elif improved and checkpoint_dir is not None:
                model_handler.save(
                    file_path=os.path.join(
                        checkpoint_dir,
                        '%02d-%.5f.pt' % (epoch, results['accu'])
                    ),
                    additional_info={'epoch': epoch, 'step': runner.step}
                )

    # adjust learning rate by epoch
    if scheduler is not None:
        scheduler.step()

logger.close()
print('Total:', time.time()-start)
print('Finished Training')
