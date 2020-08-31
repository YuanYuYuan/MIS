#!/usr/bin/env python3
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import torch
import time
import os
from training import ModelHandler

from utils import get_tty_columns, save_nifti
import yaml
from MIDP import DataLoader, DataGenerator


parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    required=True,
    help='configuration for data pipeline/model'
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

# - GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpus'])

# - model
model = ModelHandler(config['model'], checkpoint=args.checkpoint).model
model.eval()

timer = time.time()
start = timer

DL = data_gen.struct['DL']
PG = data_gen.struct['PG']
BG = data_gen.struct['BG']

# ensure the order
if PG.n_workers > 1:
    assert PG.ordered
assert BG.n_workers == 1

# Use data list from PG since may be shuffled
progress_bar = tqdm(
    zip(PG.data_list, PG.partition),
    total=len(PG.data_list),
    ncols=get_tty_columns(),
    desc='[Infering] ID: '
)

partition_counter = 0

# use with statement to make sure data_gen is clear in case interrupted
with torch.set_grad_enabled(False):
    results = torch.Tensor()
    with data_gen as gen:
        for (data_idx, partition_per_data) in progress_bar:

            # show progress
            progress_bar.set_description('[Infering] ID: %s' % data_idx)

            # loop until a partition have been covered
            while partition_counter < partition_per_data:
                try:
                    batch = next(gen)
                    logits = model(batch['image'].cuda())
                    probas = F.softmax(logits, dim=1)

                    # enhance prediction by setting a threshold
                    for i in range(1, probas.shape[1]):
                        probas[:, i, ...] += (probas[:, i, ...] >= config['output_threshold']).float()
                    output = torch.argmax(probas, 1).cpu()

                except StopIteration:
                    break

                # append to each torch tensor in results
                results = output if len(results) == 0 else torch.cat([results, output])
                partition_counter += data_gen.struct['BG'].batch_size

            # save prediction
            prediction = PG.revert(
                data_idx,
                results[:partition_per_data]
            )
            DL.save_prediction(
                data_idx,
                prediction.data.cpu().numpy(),
                args.prediction_dir
            )

            # remove processed results
            results = results[partition_per_data:]
            partition_counter -= partition_per_data


print('Total:', time.time()-start)
print('Finished')
