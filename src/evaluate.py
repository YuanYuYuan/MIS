#!/usr/bin/env python3
from tensorboardX import SummaryWriter
import argparse
import time
import os
import yaml
from training import (
    ModelHandler,
    Runner,
    SegLearner
)
from MIDP import DataLoader, DataGenerator, Reverter
from flows import MetricFlow
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
    '--log-dir',
    default='_logs',
    help='logs'
)
parser.add_argument(
    '--prediction-dir',
    default=None,
    help='save prediction',
)
args = parser.parse_args()

timer = time.time()
start = timer


if args.log_dir is not None:
    logger = SummaryWriter(args.log_dir)
else:
    logger = None

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

# check include prediction while running
if args.prediction_dir is not None or 'include_prediction' in config:
    include_prediction = True
else:
    include_prediction = False

# ensure the ability of restoration if including prediction
if include_prediction:
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
        meter=MetricFlow(config['meter']),
        optim=dict()
    ),
    logger=logger,
)

result_list = runner.run(
    data_gen,
    training=False,
    stage='Evaluating',
    include_prediction=include_prediction,
)

result_keys = [
    key for key in result_list[0].keys()
    if key not in reverter.revertible
]

# arrange the result
result = {
    key: np.nanmean(
        np.vstack([result[key] for result in result_list]),
        axis=0
    )
    for key in result_keys
}

accu = result.pop('accu')
accu_dict = {key: val for key, val in zip(ROIs, accu)}
accu_dict.update({'mean': np.mean(accu)})
print(', '.join(
    '%s: %.5f' % (key, val)
    for key, val in result.items()
))
print('Accu: ' + ', '.join(
    '%s: %.5f' % (key, val)
    for key, val in accu_dict.items()
))


scores = dict()
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

        if include_prediction:
            scores[data_idx] = DL.evaluate(data_idx, reverted['prediction'])
            if args.prediction_dir is not None:
                DL.save_prediction(
                    data_idx,
                    reverted['prediction'],
                    args.prediction_dir
                )

        else:
            scores[data_idx] = reverted['score']

        info = '[%s] ' % data_idx
        info += ', '.join(
            '%s: %.3f' % (key, val)
            for key, val in scores[data_idx].items()
        )

        progress_bar.set_description(info)

with open('score.json', 'w') as f:
    json.dump(scores, f, indent=2)

mean_roi_score = {
    roi: np.mean([
        scores[data_idx][roi]
        for data_idx in scores
    ])
    for roi in ROIs
}
mean_roi_score.update({'mean': np.mean([mean_roi_score[roi] for roi in ROIs])})
print('========== Restored ==========')
print(mean_roi_score)
print('==============================')


print('Time:', time.time()-start)
logger.close()
