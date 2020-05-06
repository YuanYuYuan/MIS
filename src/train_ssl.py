#!/usr/bin/env python3
from tensorboardX import SummaryWriter
import argparse
import time
import os
from utils import epoch_info
import yaml
from training import (
    Optimizer,
    Scheduler,
    EarlyStopper,
    ModelHandler,
    Runner,
    SegDisLearner,
    CheckpointHandler,
)
from MIDP import DataLoader, DataGenerator, Reverter
from flows import MetricFlow
import json
from tqdm import tqdm
from utils import get_tty_columns
import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    required=True,
    help='training config'
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
assert 'train' in stages
assert 'valid' in stages

# SSL
# assert 'train_ssl' in stages
if 'train_ssl' in stages:
    print('SSL is included in the training.')

with open(config['data']) as f:
    data_config = yaml.safe_load(f)
data_list = data_config['list']
loader_config = data_config['loader']

# FIXME
if 'start_adv' in config:
    start_adv = int(config['start_adv'])
else:
    start_adv = 1

if start_adv:
    print('Start adv after %d epochs.' % start_adv)


# - data pipeline
data_gen = dict()
loader_name = loader_config.pop('name')
ROIs = None
for stage in stages:
    data_loader = DataLoader(loader_name, **loader_config)
    if stage == 'train_ssl' and stage not in data_list:
        data_loader.set_data_list(data_list['valid'])
    else:
        assert stage in data_list
        data_loader.set_data_list(data_list[stage])
    data_gen[stage] = DataGenerator(data_loader, generator_config[stage])

    if ROIs is None:
        ROIs = data_loader.ROIs

# FIXME
reverter = Reverter(data_gen['valid'])

# - GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpus'])
torch.backends.cudnn.enabled = True

# - model
model_handlers = {
    key: ModelHandler(**config['models'][key])
    for key in config['models']
}

# - checkpoint handler
ckpt_handlers = {
    key: CheckpointHandler(
        model_handlers[key],
        **config['ckpt_handlers'][key]
    )
    for key in config['ckpt_handlers']
}

# - optimizer
optimizers = {
    key: Optimizer(config['optimizers'][key])(model_handlers[key].model)
    for key in ['seg', 'dis']
}

# - scheduler
if 'scheduler' in config:
    scheduler = Scheduler(optimizers['seg'], **config['scheduler'])
else:
    scheduler = None

# - load checkpoints

if args.log_dir is not None:
    logger = SummaryWriter(args.log_dir)
else:
    logger = None

timer = time.time()
start = timer

# FIXME
if 'grad_accumulation' in config:
    grad_accumulation = config['grad_accumulation']
else:
    grad_accumulation = 1
if grad_accumulation > 1:
    print('grad_accumulation:', grad_accumulation)


runner = Runner(
    learner=SegDisLearner(
        models={
            'seg': model_handlers['seg'].model,
            'dis': model_handlers['dis'].model,
        },
        optims=optimizers,
        meters={
            key: MetricFlow(config['meters'][key])
            for key in config['meters']
        },
    ),
    logger=logger
)


# early stopper
if 'early_stopper' in config:
    early_stopper = EarlyStopper(**config['early_stopper'])
else:
    early_stopper = None

# set proper initial epoch
if model_handlers['seg'].checkpoint is not None:
    init_epoch = model_handlers['seg'].checkpoint['epoch']
else:
    init_epoch = 1

# main running loop
terminated = False
scheduler_metric = None
stage_info = {
    'train': 'Training',
    'train_ssl': 'Training_SSL',
    'valid': 'Validating',
}
for epoch in range(init_epoch, init_epoch + config['epochs']):

    if terminated:
        break

    # epoch start
    epoch_info(epoch - 1, init_epoch + config['epochs'] - 1)

    for stage in stages:
        training = True if 'train' in stage else False

        # skip validation stage by validation_frequency
        if all((
            not training,
            (epoch - init_epoch) % config['validation_frequency'] != 0
        )):
            continue

        if (epoch - init_epoch) >= start_adv:
            if stage == 'train_ssl':
                mode = 'ssl'
            else:
                mode = 'adv'
        else:
            if stage == 'train_ssl':
                continue
            else:
                mode = 'normal'

        # run on an epoch
        try:
            if training:
                result_list = runner.run(
                    data_gen[stage],
                    training=training,
                    stage=stage_info[stage],
                    mode=mode,
                )
            else:
                result_list = runner.run(
                    data_gen[stage],
                    training=training,
                    stage=stage_info[stage],
                    compute_match=True,
                )

        except KeyboardInterrupt:
            print('save temporary model into %s' % args.pause_ckpt)
            # TODO: improve code
            model_handlers['seg'].save(
                args.pause_ckpt,
                additional_info={'epoch': epoch, 'step': runner.step}
            )
            model_handlers['dis'].save(
                'pause_dis.pt',
                additional_info={'epoch': epoch, 'step': runner.step}
            )
            terminated = True
            break

        # collect results except those revertible ones, e.g., accu, losses
        result = {
            key: np.nanmean(
                np.vstack([result[key] for result in result_list]),
                axis=0
            )
            for key in result_list[0].keys()
            if key not in reverter.revertible
        }

        # summarize the performance
        accu = result.pop('accu')
        accu_dict = {key: val for key, val in zip(ROIs, accu)}
        mean_accu = np.nanmean(accu)
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
                logger.add_scalar(
                    '%s/epoch/%s' % (stage_info[stage], key),
                    val,
                    epoch
                )
            logger.add_scalar(
                '%s/epoch/mean_accu' % stage_info[stage],
                mean_accu,
                epoch
            )
            logger.add_scalars(
                '%s/epoch/accu' % stage_info[stage],
                accu_dict,
                epoch
            )

        # do some stuffs depending on validation
        if stage == 'valid':

            # revert the matching dice score to the whole one from batches
            scores = dict()
            progress_bar = tqdm(
                reverter.on_batches(result_list),
                total=len(reverter.data_list),
                dynamic_ncols=True,
                ncols=get_tty_columns(),
                desc='[Data index]'
            )
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
            roi_scores = {
                roi: np.mean([
                    scores[data_idx][roi] for data_idx in scores
                ])
                for roi in ROIs
            }
            roi_scores.update({
                'mean': np.mean([
                    roi_scores[roi]
                    for roi in ROIs
                ])
            })
            print('Scores: ' + ', '.join(
                '%s: %.5f' % (key, val)
                for key, val in roi_scores.items()
            ))
            if logger is not None:
                logger.add_scalars('roi_scores', roi_scores, epoch)
                file_path = os.path.join(
                    args.log_dir,
                    '%02d-%.5f.json' % (epoch, roi_scores['mean'])
                )
                with open(file_path, 'w') as f:
                    json.dump(scores, f, indent=2)

            # update metric for learning rate scheduler
            if scheduler and scheduler.use_reduce_lr:
                if scheduler.mode == 'min':
                    scheduler_metric = result['loss']
                else:
                    scheduler_metric = roi_scores['mean']
                scheduler.step(metric=scheduler_metric)

            # check early stopping
            if early_stopper is not None:
                if early_stopper.mode == 'min':
                    early_stopper_metric = result['loss']
                else:
                    early_stopper_metric = roi_scores['mean']
                early_stop, improved = early_stopper.check(early_stopper_metric)

                if early_stop:
                    print('Early stopped.')
                    terminated = True
                    break

            # save checkpoints
            for key in ckpt_handlers:
                if ckpt_handlers[key].mode == 'each':
                    metric = roi_scores['mean']
                elif ckpt_handlers[key].criterion == 'min':
                    metric = result['loss']
                else:
                    metric = roi_scores['mean']
                ckpt_handlers[key].run(
                    metric,
                    epoch,
                    additional_info={'epoch': epoch, 'step': runner.step}
                )

    # adjust learning rate by epoch
    if scheduler and not terminated:

        if not scheduler.use_reduce_lr and stage != 'valid':
            scheduler.step()

        if logger:
            logger.add_scalar(
                'scheduler/lr_rate',
                optimizers['dis'].param_groups[0]['lr'],
                epoch
            )
            if scheduler.best is not None:
                logger.add_scalar(
                    'scheduler/best_metric',
                    scheduler.best,
                    epoch
                )
                logger.add_scalar(
                    'scheduler/n_stagnation',
                    scheduler.n_stagnation,
                    epoch
                )

logger.close()
print('Total:', time.time()-start)
print('Finished Training')
