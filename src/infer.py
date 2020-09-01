#!/usr/bin/env python3
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
    '--prediction-dir',
    default=None,
    help='save prediction',
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
model_handler = ModelHandler(config['model'], checkpoint=args.checkpoint)
runner = Runner(
    learner=SegLearner(
        model=model_handler.model,
        meter=None,
        optim=dict()
    ),
    logger=None,
)

result_list = runner.run(
    data_gen,
    training=False,
    stage='Evaluating',
    include_prediction=True,
    compute_match=False,
)

for key in result_list[0].keys():
    assert key in reverter.revertible, f'{key} is not revertible!'

with tqdm(
    reverter.on_batches(
        result_list,
        output_threshold=config['output_threshold']
    ),
    total=len(reverter.data_list),
    dynamic_ncols=True,
    ncols=get_tty_columns(),
    desc='[Data index]'
) as progress_bar:
    for reverted in progress_bar:
        data_idx = reverted['idx']
        DL.save_prediction(
            data_idx,
            reverted['prediction'],
            args.prediction_dir
        )

        info = '[%s] ' % data_idx
        progress_bar.set_description(info)

print('Time:', time.time()-start)
