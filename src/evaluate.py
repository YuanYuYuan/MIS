#!/usr/bin/env python3
import argparse
import time
import os
import yaml
from training.model_handler import ModelHandler
from training.runner import Runner
from MIDP import DataLoader, DataGenerator
from flows import MetricFlow


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
data_gen = dict()
loader_name = loader_config.pop('name')
data_loader = DataLoader(loader_name, **loader_config)
if data_list is not None:
    data_loader.set_data_list(data_list)
data_gen = DataGenerator(data_loader, generator_config)

ROIs = None
if ROIs is None:
    ROIs = data_loader.ROIs

# - GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpus'])

# - model
model_handler = ModelHandler(config['model'], checkpoint=args.checkpoint)
runner = Runner(
    model=model_handler.model,
    meter=MetricFlow(config['meter']),
)
results = runner.run(data_gen, training=False)
print(', '.join(
    '%s: %.5f' % (key, val)
    for key, val in results.items()
))
print('Total:', time.time()-start)
print('Finished Training')
