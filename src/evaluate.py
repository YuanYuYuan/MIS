#!/usr/bin/env python3
from tensorboardX import SummaryWriter
import argparse
import time
import os
import yaml
from training.model_handler import ModelHandler
from training.runner import Runner
from MIDP import DataLoader, DataGenerator
from flows import MetricFlow
from tqdm import tqdm
import numpy as np
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

ROIs = None
if ROIs is None:
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
    model=model_handler.model,
    meter=MetricFlow(config['meter']),
    logger=logger,
)

# if include_prediction:
#     result_list, prediction_list = runner.run(
#         data_gen,
#         training=False,
#         stage='Evaluating',
#         include_prediction=True,
#     )

#     assert len(prediction_list) * BG.batch_size >= sum(PG.partition), \
#         (len(prediction_list) * BG.batch_size, sum(PG.partition))

#     def dice_score(x, y):
#         assert x.shape == y.shape
#         return 2 * np.sum(x * y) / np.sum(x + y)

#     progress_bar = tqdm(
#         zip(PG.data_list, PG.partition),
#         total=len(PG.data_list),
#         dynamic_ncols=True,
#         desc='[Data index]'
#     )

#     queue = []
#     scores = dict()
#     for (data_idx, partition_per_data) in progress_bar:

#         # collect new batch prediction into queue
#         while len(queue) < partition_per_data:
#             batch = prediction_list.pop(0)
#             if len(queue) == 0:
#                 queue = batch
#             else:
#                 queue = np.concatenate((queue, batch), axis=0)

#         # restore if the queue is enough for restoration
#         restored = PG.restore(
#             data_idx,
#             queue[:partition_per_data],
#             output_threshold=config['output_threshold'],
#         )

#         # clean out restored part
#         queue = queue[partition_per_data:]

#         # collect score for each data
#         scores[data_idx] = {
#             roi: dice_score(
#                 (restored == val).astype(int),
#                 (DL.get_label(data_idx) == val).astype(int)
#             )
#             for roi, val in DL.roi_map.items()
#         }

#         # update progress bar
#         info = '[%s] ' % data_idx
#         info += ', '.join(
#             '%s: %.3f' % (key, val)
#             for key, val in scores[data_idx].items()
#         )
#         progress_bar.set_description(info)

#         # save prediction if specified
#         if args.prediction_dir is not None:
#             DL.save_prediction(
#                 data_idx,
#                 restored,
#                 args.prediction_dir
#             )

#     with open('score.json', 'w') as f:
#         json.dump(scores, f, indent=2)

#     mean_roi_score = {
#         roi: np.mean([scores[key][roi] for key in scores])
#         for roi in ROIs
#     }
#     mean_roi_score.update({'mean': np.mean([mean_roi_score[roi] for roi in ROIs])})
#     print('========== Restored ==========')
#     print(mean_roi_score)
#     print('==============================')

# else:
#     result_list = runner.run(
#         data_gen,
#         training=False,
#         stage='Evaluating',
#         include_prediction=False,
#     )

result_list = runner.run(
    data_gen,
    training=False,
    stage='Evaluating',
    include_prediction=include_prediction,
    compute_match=False,
)


result_keys = list(result_list[0].keys())
if 'prediction' in result_keys:
    result_keys.remove('prediction')
    print('do something on prediction')

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

print('Time:', time.time()-start)
logger.close()
