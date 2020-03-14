#!/usr/bin/env python3
from tensorboardX import SummaryWriter
import argparse
import time
import os
from utils import Runner, epoch_info, EarlyStopper, ModelHandler
import yaml
from training.optimizers import Optimizer
from training.scheduler import CosineAnnealingWarmUpRestarts as Scheduler
from MIDP import DataLoader, DataGenerator
# import models


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

# - loss
if 'losses' in config['loss']:
    from training.losses import MixedLoss
    loss_fn = MixedLoss(config['loss'])
else:
    from training.losses import Loss
    loss_fn = Loss(config['loss'])

if args.log_dir is not None:

    # TensorboardX logging
    logger = SummaryWriter(args.log_dir)

    # ROIScoreWriter
    # score_writer = ROIScoreWriter(
    #     os.path.join(args.log_dir, 'scores.csv'),
    #     ROIs
    # )

else:
    logger = None

timer = time.time()
start = timer

runners = {
    'train': Runner(
        model_handler.model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        logger=logger,
    ),
    'valid': Runner(
        model_handler.model,
        loss_fn=loss_fn,
        logger=logger,
    )
}

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
for epoch in range(init_epoch, init_epoch + config['epochs']):

    # epoch start
    epoch_info(epoch, init_epoch + config['epochs'])

    for stage in ['train', 'valid']:

        if stage == 'valid' and epoch % config['validation_frequency'] != 0:
            break

        # run on an epoch
        try:
            results = runners[stage].run(data_gen[stage])
        except KeyboardInterrupt:
            print('save temporary model into %s' % args.pause_ckpt)
            model_handler.save(
                args.pause_ckpt,
                additional_info={
                    'epoch': epoch,
                    'step': runners[stage].step
                }
            )
            break

        # summarize the performance
        info = ''
        for key, val in results.item():
            info += '%s: %.5f' % (key, val)
        print(info)

        # record the performance
        if logger is not None:
            for key, val in results.item():
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
                break

            elif improved and checkpoint_dir is None:
                model_handler.save(
                    file_path=os.path.join(
                        checkpoint_dir,
                        '%02d-%.5f.pt' % (epoch, results['accu'])
                    ),
                    additional_info={
                        'epoch': epoch,
                        'step': runners[stage].step
                    }
                )

    # adjust learning rate by epoch
    if scheduler is not None:
        scheduler.step()

logger.close()
print('Total:', time.time()-start)
print('Finished Training')
