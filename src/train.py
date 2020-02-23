#!/usr/bin/env python3
from tensorboardX import SummaryWriter
import argparse
import torch
import time
import os

from utils import Trainer, Predictor, epoch_info, ROIScoreWriter
import yaml
from training.optimizers import Optimizer
from training.scheduler import CosineAnnealingWarmUpRestarts as Scheduler
from MIDP import DataLoader, DataGenerator
import models


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
ROIs = loader_config[0]['ROIs']
for stage in ['train', 'valid']:
    data_loader = DataLoader(*loader_config)
    if data_list[stage] is not None:
        data_loader.set_data_list(data_list[stage])
    data_gen[stage] = DataGenerator(data_loader, generator_config[stage])

# - GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpus'])

# - model
model_name = config['model'].pop('name')
model = getattr(models, model_name)(**config['model'])

# - optimizer
optimizer = Optimizer(config['optimizer'])(model)

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
    score_writer = ROIScoreWriter(
        os.path.join(args.log_dir, 'scores.csv'),
        ROIs
    )

else:
    logger = None

timer = time.time()
start = timer

trainer = Trainer(
    model,
    optimizer,
    loss_fn=loss_fn,
    accu_fn='dice',
    load_checkpoint=args.checkpoint,
    logger=logger,
)

predictor = Predictor(
    trainer.model,
    threshold=config['output_threshold'],
)

checkpoint_dir = args.checkpoint_dir
if checkpoint_dir:
    os.makedirs(checkpoint_dir, exist_ok=True)

early_stop = config['early_stopping_epochs'] > 1
best_score = 0.0
n_stagnation = 0

for epoch in range(config['epochs']):

    try:

        # epoch start
        epoch += trainer.init_epoch
        epoch_info(epoch, config['epochs'] + trainer.init_epoch)

        # train an epoch
        loss, accu = trainer.run(data_gen['train'], epoch)

        # adjust learning rate by epoch
        if scheduler is not None:
            scheduler.step()

        # epoch summary
        print('Avg Loss: %.5f, Avg Accu: %.5f' % (loss, accu))
        if logger is not None:
            logger.add_scalar('metrics/epoch_loss', loss, epoch+1)
            logger.add_scalar('metrics/epoch_accu', accu, epoch+1)

        if epoch % config['validation_frequency'] == 0:

            # get validation scorel: {data_idx: {roi: score}}
            val_score = predictor.run(data_gen['valid'])

            # compute total average score
            avg_score = sum(
                sum(val_score[data_idx].values()) / len(val_score[data_idx])
                for data_idx in val_score
            ) / len(val_score)

            # compute score for each roi: {roi: average score}
            roi_score = {roi: 0.0 for roi in ROIs}
            for data_idx in val_score.keys():
                for roi in ROIs:
                    roi_score[roi] += val_score[data_idx][roi]
            for roi in ROIs:
                roi_score[roi] /= len(val_score)

            info = ['Avg Accu: %.5f' % avg_score]
            info += [
                '%s: %.5f' % (roi, score)
                for (roi, score) in roi_score.items()
            ]
            print(', '.join(info))

            # logging
            if logger is not None:
                logger.add_scalar(
                    'validation/avg_accu',
                    avg_score,
                    epoch+1
                )
                logger.add_scalars(
                    'validation/roi_score',
                    roi_score,
                    epoch+1
                )

                # log validation score into csv file
                score_writer.write(epoch+1, roi_score)

            # check early stopping
            if early_stop:
                if avg_score > best_score:
                    best_score = avg_score
                    n_stagnation = 0

                    # store checkpoint
                    if checkpoint_dir:
                        file_path = os.path.join(
                            checkpoint_dir,
                            '%02d-%.5f.pt' % (epoch+1, avg_score)
                        )

                        # check if multiple gpu model
                        if torch.cuda.device_count() > 1:
                            model_state_dict = trainer.model.module.state_dict()
                        else:
                            model_state_dict = trainer.model.state_dict()

                        torch.save({
                            'epoch': epoch+1,
                            'model_state_dict': model_state_dict,
                            'loss': loss,
                            'step': trainer.global_step
                        }, file_path)
                elif avg_score > best_score * 0.95:
                    continue
                else:
                    n_stagnation += 1
                    if n_stagnation > config['early_stopping_epochs']:
                        print('Early stop.')
                        break

    except KeyboardInterrupt:

        # check if multiple gpu model
        if torch.cuda.device_count() > 1:
            model_state_dict = trainer.model.module.state_dict()
        else:
            model_state_dict = trainer.model.state_dict()

        # save temporary model into pause.pt
        print('save temporary model into %s' % args.pause_ckpt)
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model_state_dict,
            'step': trainer.global_step
        }, args.pause_ckpt)

        break

logger.close()
print('Total:', time.time()-start)
print('Finished Training')
