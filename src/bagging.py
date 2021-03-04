#!/usr/bin/env python3
import torch.nn.functional as F
import argparse
import time
import os
import yaml
from training import ModelHandler, Runner, SegLearner
from MIDP import DataLoader, DataGenerator, Reverter
from tqdm import tqdm
import numpy as np
from utils import get_tty_columns
import torch
import json
import nrrd


parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    required=True,
    help='training config'
)
parser.add_argument(
    '--checkpoints',
    nargs='+',
    default=None,
    help='pretrained model checkpoint'
)
parser.add_argument(
    '--prediction-dir',
    default='outputs',
    help='save prediction',
)
parser.add_argument(
    '--keep-ch',
    default=False,
    action='store_true'
)
args = parser.parse_args()

timer = time.time()
start = timer



# load config
with open(args.config) as f:
    config = yaml.safe_load(f)
generator_config = config['generator']
with open(config['data']) as f:
    data_config = yaml.safe_load(f)
data_list = data_config['list']
loader_config = data_config['loader']

# - data pipeline
loader_name = loader_config.pop('name')
data_loader = DataLoader(loader_name, **loader_config)
if data_list is not None:
    data_loader.set_data_list(data_list)
data_gen = DataGenerator(data_loader, generator_config)
reverter = Reverter(data_gen)

ROIs = data_loader.ROIs
DL = data_gen.struct['DL']
PG = data_gen.struct['PG']
BG = data_gen.struct['BG']
# ensure the order
if PG.n_workers > 1:
    assert PG.ordered
assert BG.n_workers == 1
if 'AG' in data_gen.struct:
    assert data_gen.struct['AG'].n_workers == 1

assert 'output_threshold' in config

if args.prediction_dir is not None:
    os.makedirs(args.prediction_dir, exist_ok=True)

# - GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpus'])

# - model
print(args.checkpoints)
model_handlers = [
    ModelHandler(config['model'], checkpoint=ckpt)
    for ckpt in args.checkpoints
]

result = []

n_models = len(model_handlers)
batches = []
for batch in tqdm(data_gen):
    for key in batch:
        batch[key] = batch[key].cuda()

    prob = None
    for mh in model_handlers:
        with torch.set_grad_enabled(False):
            mh.model.eval()
            pred = mh.model(batch)['prediction']
            if prob is None:
                prob = F.softmax(pred, dim=1).detach().cpu().numpy()
            else:
                prob += F.softmax(pred, dim=1).detach().cpu().numpy()

    prob /= n_models
    batches.append({'prediction': prob})


for key in batches[0].keys():
    assert key in reverter.revertible, f'{key} is not revertible!'

with tqdm(
    reverter.on_batches(
        batches,
        output_threshold=config['output_threshold']
    ),
    total=len(reverter.data_list),
    dynamic_ncols=True,
    ncols=get_tty_columns(),
    desc='[Data index]'
) as progress_bar:
    for reverted in progress_bar:
        data_idx = reverted['idx']

        if len(reverted['prediction'].shape) > 3:
            nrrd.write(
                os.path.join(
                    args.prediction_dir,
                    data_idx + '.nrrd',
                ),
                reverted['prediction'].astype('float32'),
                compression_level=5,
            )

        else:
            DL.save_prediction(
                data_idx,
                reverted['prediction'],
                args.prediction_dir
            )

        info = '[%s] ' % data_idx
        progress_bar.set_description(info)

print('Time:', time.time()-start)
