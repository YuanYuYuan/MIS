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
from multiprocessing import Pool


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
    help='save prediction'
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

if 'save_prediction' in config:
    PG = data_gen.struct['PG']
    BG = data_gen.struct['BG']
    # ensure the order
    if PG.n_workers > 1:
        assert PG.ordered
    assert BG.n_workers == 1
    prediction_dir = config['save_prediction']
    os.makedirs(prediction_dir, exist_ok=True)
    save_prediction = True
else:
    save_prediction = False


# - GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpus'])

# - model
model_handler = ModelHandler(config['model'], checkpoint=args.checkpoint)
runner = Runner(
    model=model_handler.model,
    meter=MetricFlow(config['meter']),
    logger=logger,
)
result_list = runner.run(
    data_gen,
    training=False,
    stage='Evaluating',
    save_prediction=save_prediction,
)
result_keys = list(result_list[0].keys())

if save_prediction:
    assert 'prediction' in result_keys

    def save_npy(data):
        idx, npy = data
        file_path = os.path.join(prediction_dir, ('%03d' % idx) + '.npy')
        np.save(file_path, npy)

    with Pool(4) as pool:
        jobs = pool.imap(save_npy, enumerate(result_list))
        list(tqdm(
            jobs,
            total=len(result_list),
            dynamic_ncols=False,
            desc='[Saving prediction]',
        ))

result_keys.remove('prediction')
result = {
    key: torch.stack([result[key] for result in result_list]).mean(dim=0)
    for key in result_keys
}

accu = result.pop('accu')
accu_dict = {key: val.item() for key, val in zip(ROIs, accu)}
accu_dict.update({'mean': accu.mean()})
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
