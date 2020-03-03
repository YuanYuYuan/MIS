#!/usr/bin/env python3
from tensorboardX import SummaryWriter
import argparse
import torch
import time
import os

from utils import Trainer, Validator, epoch_info, ROIScoreWriter, EarlyStopper
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

validator = Validator(
    trainer.model,
    threshold=config['output_threshold'],
)

if args.validate_only:
    validator.run(data_gen['valid'])
    logger.close()
    print('Total:', time.time()-start)
    exit(0)

checkpoint_dir = args.checkpoint_dir
if checkpoint_dir:
    os.makedirs(checkpoint_dir, exist_ok=True)

# early stopper
if config['early_stopping_epochs'] > 1:
    early_stopper = EarlyStopper(config['early_stopping_epochs'])
else:
    early_stopper = None

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

            # get validation score
            val_score = validator.run(data_gen['valid'])

            # logging
            if logger is not None:
                logger.add_scalars(
                    'validation/roi_score',
                    val_score['roi'],
                    epoch+1
                )
                logger.add_scalar(
                    'validation/avg_accu',
                    val_score['avg'],
                    epoch+1
                )

                # log validation score into csv file
                score_writer.write(epoch+1, val_score['roi'])

            # check early stopping
            if early_stopper:
                early_stop, improved = early_stopper.check(val_score['avg'])

                if early_stop:
                    print('Early stopped.')
                    break

                elif improved:
                    # store checkpoint
                    if checkpoint_dir:
                        file_path = os.path.join(
                            checkpoint_dir,
                            '%02d-%.5f.pt' % (epoch+1, val_score['avg'])
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
