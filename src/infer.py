#!/usr/bin/env python3
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import torch
import time
import os

from utils import get_tty_columns, save_nifti
import yaml
from MIDP import DataLoader, DataGenerator
import models


parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    required=True,
    help='configuration for data pipeline/model'
)
args = parser.parse_args()


# load config
with open(args.config) as f:
    config = yaml.safe_load(f)

loader_config = config['loader']

# - data pipeline
loader_name = loader_config.pop('name')
data_loader = DataLoader(
    loader_name,
    **loader_config
)

data_gen = DataGenerator(data_loader, config['generator'])
ROIs = data_loader.ROIs

# - GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpus'])

# - model
model_name = config['model'].pop('name')
model = getattr(models, model_name)(**config['model'])
print('===== Loading checkpoint %s =====' % config['model_weight'])
checkpoint = torch.load(
    config['model_weight'],
    map_location=lambda storage,
    location: storage
)
model.load_state_dict(checkpoint['model_state_dict'])
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model).cuda()
else:
    model = model.cuda()
model.eval()

timer = time.time()
start = timer

PG = data_gen.struct['PG']
DL = data_gen.struct['DL']
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
    desc='[Predicting] ID: '
)

partition_counter = 0

# use with statement to make sure data_gen is clear in case interrupted
with data_gen as gen:
    for (data_idx, partition_per_data) in progress_bar:

        results = []

        # loop until a partition have been covered
        while partition_counter < partition_per_data:
            try:
                batch = next(gen)
                logits = model(batch['image'].cuda())
                probas = F.softmax(logits, dim=1)

                # enhance prediction by setting a threshold
                for i in range(1, probas.shape[1]):
                    probas[:, i, ...] += (probas[:, i, ...] >= config['output_threshold']).float()

                # convert probabilities into one-hot: [B, C, ...]
                max_idx = torch.argmax(probas, 1, keepdim=True)
                one_hot = torch.zeros(probas.shape)
                one_hot.scatter_(1, max_idx, 1)
                results.append(one_hot)

            except StopIteration:
                break

            # append to each torch tensor in results list
            partition_counter += data_gen.struct['BG'].batch_size

        # save prediction
        one_hot_ouptuts = results[:partition_per_data]
        prediction = PG.restore(
            data_idx,
            torch.argmax(one_hot_ouptuts, 1)
        )
        save_nifti(
            prediction.data.cpu().numpy(),
            os.path.join(
                config['output_dir'],
                data_idx + '.nii.gz'
            )
        )

        # remove processed results
        results = results[partition_per_data:]
        partition_counter -= partition_per_data

        # show progress
        progress_bar.set_description('[Predicting]  ID: %s' % data_idx)

print('Total:', time.time()-start)
print('Finished')
