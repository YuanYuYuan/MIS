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
    DetLearner
)
import torch.nn.functional as F
from models import Classifier
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
for stage in stages:
    data_loader = DataLoader(loader_name, **loader_config)
    if data_list[stage] is not None:
        data_loader.set_data_list(data_list[stage])
    data_gen[stage] = DataGenerator(data_loader, generator_config[stage])

# # - GPUs
# os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpus'])
# torch.backends.cudnn.enabled = True

# - model
model = Classifier(out_channels=2)
model.zero_grad()

# - optimizer
optim = Optimizer(config['optimizer'])(model)
optim.zero_grad()

weight = torch.Tensor([0.001, 0.999])
# weight = torch.Tensor([0.5, 0.5])
# weight = torch.Tensor([0.99, 0.01])
criterion = torch.nn.CrossEntropyLoss(weight=weight)
def F1_score(predis, labels):
    return 2 * torch.sum(predis * labels) / torch.sum(predis + labels)

def precision(predis, labels):
    TP = 2 * torch.sum(predis * labels)
    FP = torch.sum(labels) - TP
    FN = torch.sum(predis) - TP
    return TP / (TP + FP)

timer = time.time()
start = timer

checkpoint_dir = args.checkpoint_dir
if checkpoint_dir:
    os.makedirs(checkpoint_dir, exist_ok=True)

init_epoch = 1

# main running loop
terminated = False
scheduler_metric = None
stage_info = {'train': 'Training', 'valid': 'Validating'}
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
            n_steps = len(data_gen[stage])
            progress_bar = tqdm(
                data_gen[stage],
                total=n_steps,
                ncols=get_tty_columns(),
                dynamic_ncols=True,
                desc='[%s] Loss: %.5f, Accu: %.5f'
                % (stage_info[stage], 0.0, 0.0)
            )

            avg_loss = 0.0
            avg_accu = 0.0
            union = 0.0
            match = 0.0
            counter = 0

            training = stage == 'train'
            if training:
                model.train()
            else:
                model.eval()

            for data in progress_bar:
                labels = (torch.sum(data['label'], dim=(1, 2, 3)) > 0).long()
                # labels = torch.unsqueeze(labels, dim=1).float()

                images = data['image']
                if torch.sum(images) > 0:

                    # label = label.cuda()
                    # image = image.cuda()

                    with torch.set_grad_enabled(training):

                        logits = model(images)
                        loss = criterion(logits, labels)
                        predis = torch.argmax(logits, dim=1).float()
                        accu = 1 - F.mse_loss(predis, labels)
                        # accu = precision(predis, labels)

                        union += torch.sum(predis + labels)
                        match += torch.sum(predis * labels)

                        if training:
                            loss.backward()
                            optim.step()
                            model.zero_grad()

                    avg_loss += loss.detach().cpu().item()
                    avg_accu += accu.detach().cpu().item()
                    counter += 1

                else:
                    loss = -1
                    accu = -1

                progress_bar.set_description(
                    '[%s] Running loss: %.5f, accu: %.5f'
                    % (stage_info[stage], loss, accu)
                )


            dice = 2 * match / union
            print(
                '[%s] Avg loss: %.5f, accu: %.5f, dice: %.5f'
                % (stage_info[stage], avg_loss / counter, avg_accu / counter, dice)
            )

        except KeyboardInterrupt:
            print('save temporary model into %s' % args.pause_ckpt)
            terminated = True
            break

print('Total:', time.time()-start)
print('Finished Training')
