#!/usr/bin/env python3
from tensorboardX import SummaryWriter
import argparse
import time
import os
from utils import epoch_info, EarlyStopper
import yaml
from training.optimizers import Optimizer
from training.scheduler import Scheduler
from training.model_handler import ModelHandler
from training.runner import Runner
from MIDP import DataLoader, DataGenerator, Reverter
from flows import MetricFlow
import torch
from tqdm import tqdm
from utils import get_tty_columns
import numpy as np


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
    '--pause-ckpt',
    help='save model checkpoint if paused'
)
args = parser.parse_args()

# load config
with open(args.config) as f:
    config = yaml.safe_load(f)
generator_config = config['generator']
stages = generator_config.keys()
with open(config['data']) as f:
    data_config = yaml.safe_load(f)
data_list = data_config['list']
loader_config = data_config['loader']

# - data pipeline
data_gen = dict()
loader_name = loader_config.pop('name')
ROIs = None
for stage in stages:
    data_loader = DataLoader(loader_name, **loader_config)
    if data_list[stage] is not None:
        data_loader.set_data_list(data_list[stage])
    data_gen[stage] = DataGenerator(data_loader, generator_config[stage])

    if ROIs is None:
        ROIs = data_loader.ROIs

# FIXME
reverter = Reverter(data_gen['valid'])

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
        **config['scheduler']
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
scheduler_metric = None
for epoch in range(init_epoch, init_epoch + config['epochs']):

    if terminated:
        break

    # epoch start
    epoch_info(epoch - 1, init_epoch + config['epochs'] - 1)

    for stage in stages:
        training = True if stage == 'train' else False

        # skip validation stage by validation_frequency
        if stage != 'train' and epoch % config['validation_frequency'] != 0:
            break

        # run on an epoch
        try:
            result_list = runner.run(
                data_gen[stage],
                training=training,
                stage='Training' if training else 'Validating'
            )

        except KeyboardInterrupt:
            print('save temporary model into %s' % args.pause_ckpt)
            model_handler.save(
                args.pause_ckpt,
                additional_info={'epoch': epoch, 'step': runner.step}
            )
            terminated = True
            break

        # collect results except those revertible ones,
        # e.g., prediction, match dice score, ...
        result = {
            key: torch.stack([
                result[key] for result in result_list
            ]).mean(dim=0)
            for key in result_list[0].keys()
            if key not in reverter.revertible
        }

        # summarize the performance
        accu = result.pop('accu')
        accu_dict = {key: val.item() for key, val in zip(ROIs, accu)}
        mean_accu = accu.mean()
        accu_dict.update({'mean': mean_accu})
        print(', '.join(
            '%s: %.5f' % (key, val)
            for key, val in result.items()
        ))
        print('Accu: ' + ', '.join(
            '%s: %.5f' % (key, val)
            for key, val in accu_dict.items()
        ))

        # record the performance
        if logger is not None:
            for key, val in result.items():
                logger.add_scalar('%s/epoch/%s' % (stage, key), val, epoch)
            logger.add_scalar('%s/epoch/mean_accu' % stage, mean_accu, epoch)
            logger.add_scalars('%s/epoch/accu' % stage, accu_dict, epoch)

        # do some stuffs depending on validation
        if stage == 'valid':

            # revert the matching dice score to the whole one from batches
            scores = dict()
            with tqdm(
                reverter.on_batches(result_list),
                total=len(reverter.data_list),
                dynamic_ncols=True,
                ncols=get_tty_columns(),
                desc='[Data index]'
            ) as progress_bar:
                for reverted in progress_bar:
                    data_idx = reverted['idx']
                    scores[data_idx] = reverted['score']
                    info = '[%s] ' % data_idx
                    info += ', '.join(
                        '%s: %.3f' % (key, val)
                        for key, val in scores[data_idx].items()
                    )
                    progress_bar.set_description(info)

            # summerize roi score
            mean_roi_score = {
                roi: np.mean([scores[key][roi] for key in scores])
                for roi in ROIs
            }
            mean_roi_score.update({
                'mean': np.mean([
                    mean_roi_score[roi]
                    for roi in ROIs
                ])
            })
            if logger is not None:
                logger.add_scalars('mean_roi_score', mean_roi_score, epoch)

            # update metric for learning rate scheduler
            if scheduler and scheduler.use_reduce_lr:
                if scheduler.mode == 'min':
                    scheduler_metric = result['loss']
                else:
                    scheduler_metric = mean_roi_score['mean']

            # check early stopping
            if early_stopper is not None:
                early_stop, improved = early_stopper.check(mean_roi_score)

                if early_stop:
                    print('Early stopped.')
                    terminated = True
                    break

                elif improved and checkpoint_dir is not None:
                    model_handler.save(
                        file_path=os.path.join(
                            checkpoint_dir,
                            '%02d-%.5f.pt' % (epoch, mean_roi_score['mean'])
                        ),
                        additional_info={'epoch': epoch, 'step': runner.step}
                    )

    # adjust learning rate by epoch
    if scheduler:

        if scheduler.use_reduce_lr and stage == 'valid' and scheduler_metric:
            scheduler.step(metric=scheduler_metric)
        else:
            scheduler.step()

        if logger:
            logger.add_scalar(
                'scheduler/lr_rate',
                optimizer.param_groups[0]['lr'],
                epoch
            )
            if scheduler.best is not None:
                logger.add_scalar(
                    'scheduler/best_metric',
                    scheduler.best,
                    epoch
                )

logger.close()
print('Total:', time.time()-start)
print('Finished Training')
