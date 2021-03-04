#!/usr/bin/env python3
import argparse
import json5
import time
import os
from training import ModelHandler
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
    help='inference config'
)
parser.add_argument(
    '--output',
    default='./outputs',
    help='output directory',
)
parser.add_argument(
    '--test',
    default=False,
    action='store_true',
    help='check for feasibility',
)
args = parser.parse_args()
os.makedirs(args.output, exist_ok=True)

timer = time.time()
start = timer

# load config
with open(args.config) as f:
    config = json5.load(f)

# build up the data generator
with open(config['generator']['data']) as f:
    data_config = json5.load(f)
data_list = data_config['list']
if args.test:
    data_list = data_list[:1]
loader_config = data_config['loader']
loader_name = loader_config.pop('name')
data_loader = DataLoader(loader_name, **loader_config)
data_loader.set_data_list(data_list)
data_gen = DataGenerator(data_loader, config['generator']['struct'])

# build up the reverter
reverter = Reverter(data_gen)
DL = data_gen.struct['DL']
PG = data_gen.struct['PG']
BG = data_gen.struct['BG']
# ensure the order
if PG.n_workers > 1:
    assert PG.ordered
assert BG.n_workers == 1
if 'AG' in data_gen.struct:
    assert data_gen.struct['AG'].n_workers == 1

# - GPUs
if 'gpus' in config:
    if isinstance(config['gpus'], list):
        gpus = ','.join([str(idx) for idx in config['gpus']])
    else:
        assert isinstance(config['gpus'], str)
        with open(config['gpus']) as f:
            gpus = f.read().strip()
else:
    gpus = ''
if len(gpus) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.backends.cudnn.enabled = True

# modules
model_handlers = dict()
for (key, cfg) in config['module'].items():
    if 'ckpt' in cfg:
        ckpt = cfg['ckpt']
    else:
        ckpt = None
    model_handlers[key] = ModelHandler(cfg['config'], checkpoint=ckpt)
    model_handlers[key].model.eval()

    # toggle off all trainable parameters of each module
    for param in model_handlers[key].model.parameters():
        param.requires_grad = False


progress_bar = tqdm(
    data_gen,
    total=len(data_gen),
    ncols=get_tty_columns(),
    dynamic_ncols=True,
    desc='[Inferring]'
)

result_list = []
for batch in progress_bar:
    assert isinstance(batch, dict)
    data = dict()
    for key in batch:
        if torch.cuda.device_count() >= 1:
            data[key] = batch[key].cuda()
        else:
            data[key] = batch[key]

    with torch.set_grad_enabled(False):
        for mod in config['infer']['forward']:
            data.update(model_handlers[mod].model(data))

    assert 'prediction' in data
    result_list.append({
        'prediction': data['prediction'].detach().cpu().numpy(),
    })

# sanity check
for key in result_list[0].keys():
    assert key in reverter.revertible, f'{key} is not revertible!'

with tqdm(
    reverter.on_batches(
        result_list,
        output_threshold=config['infer']['threshold']
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
                    args.output,
                    data_idx + '.nrrd',
                ),
                reverted['prediction'].astype('float32'),
                compression_level=5,
            )

        else:
            DL.save_prediction(
                data_idx,
                reverted['prediction'],
                args.output
            )

        info = '[%s] ' % data_idx
        progress_bar.set_description(info)

print('Time:', time.time()-start)
