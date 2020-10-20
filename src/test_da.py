#!/usr/bin/env python3
import json5
import json
from tensorboardX import SummaryWriter
from training import ModelHandler, Optimizer
from flows import MetricFlow, ModuleFlow
from tqdm import tqdm
from utils import get_tty_columns, epoch_info
from metrics import match_up
import torch
import math
import numpy as np
from MIDP import DataLoader, DataGenerator, Reverter
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log-dir',
    default='_logs',
    help='training logs'
)
parser.add_argument(
    '--ckpt-dir',
    default='_ckpts',
    help='training checkpoints'
)
args = parser.parse_args()

logger = SummaryWriter(args.log_dir)
os.makedirs(args.ckpt_dir, exist_ok=True)


class Trainer:

    def __init__(self, config, logger=None):

        self.logger = logger
        self.step = dict()
        self.config = config

        # setup models and optimzers
        self.handlers = dict()
        self.optims = dict()
        for (key, cfg) in self.config['module'].items():
            if 'ckpt' in cfg:
                ckpt = cfg['ckpt']
            else:
                ckpt = None
            self.handlers[key] = ModelHandler(
                cfg['config'],
                checkpoint=ckpt,
            )
            self.optims[key] = Optimizer(cfg['optim'])(self.handlers[key].model)
            self.optims[key].zero_grad()

        self.metrics = {
            key: MetricFlow(config) for (key, config)
            in self.config['metric'].items()
        }

        # setup data generators
        self.generators = dict()
        for (key, cfg) in self.config['generator'].items():
            with open(cfg['data']) as f:
                data_config = json5.load(f)
            data_list = data_config['list']
            loader_config = data_config['loader']
            loader_name = loader_config.pop('name')
            data_loader = DataLoader(loader_name, **loader_config)
            data_loader.set_data_list(data_list)
            self.generators[key] = DataGenerator(data_loader, cfg['struct'])

    def run(self, stage):
        stage_config = self.config['stage'][stage]

        if len(stage_config['generator']) > 1:
            data_gen = zip(*[self.generators[gen] for gen in stage_config['generator']])
        else:
            data_gen = self.generators[stage_config['generator'][0]]
        class_names = self.generators[stage_config['generator'][0]].struct['DL'].ROIs

        n_steps = min([len(self.generators[gen]) for gen in stage_config['generator']])
        progress_bar = tqdm(
            data_gen,
            total=n_steps,
            ncols=get_tty_columns(),
            dynamic_ncols=True,
            desc='[%s] loss: %.5f, accu: %.5f'
            % (stage, 0.0, 0.0)
        )

        if stage not in self.step:
            self.step[stage] = 1

        # toggle trainable parameters of each module
        need_backward = False
        for key, toggle in stage_config['toggle'].items():
            self.handlers[key].model.train(toggle)
            for param in self.handlers[key].model.parameters():
                param.requires_grad = toggle
            if toggle:
                need_backward = True

        result_list = []
        for batch in progress_bar:

            self.step[stage] += 1

            # prepare data, merge batch from multiple generator if needed
            data = dict()
            if isinstance(batch, tuple):
                for key in batch[0]:
                    if torch.cuda.device_count() >= 1:
                        data[key] = torch.cat([sub_batch[key] for sub_batch in batch]).cuda()
                    else:
                        data[key] = torch.cat([sub_batch[key] for sub_batch in batch])

            else:
                assert isinstance(batch, dict)
                for key in batch:
                    if torch.cuda.device_count() >= 1:
                        data[key] = batch[key].cuda()
                    else:
                        data[key] = batch[key]

            # feed in data and run
            for key in stage_config['forward']:
                data.update(self.handlers[key].model(data))

            # compute loss and accuracy
            results = self.metrics[stage_config['metric']](data)

            # backpropagation
            if need_backward:
                results['loss'].backward()
                for key, toggle in stage_config['toggle'].items():
                    if toggle:
                        self.optims[key].step()
                        self.optims[key].zero_grad()

            # compute match for dice score of each case after reversion
            if stage_config['revert']:
                assert 'prediction' in data, list(data.keys())
                assert 'label' in data, list(data.keys())
                with torch.set_grad_enabled(False):
                    match, total = match_up(
                        data['prediction'],
                        data['label'],
                        needs_softmax=True,
                        batch_wise=True,
                        threshold=-1,
                    )
                    results.update({'match': match, 'total': total})

            # detach all results, move to CPU, and convert to numpy
            for key in results:
                results[key] = results[key].detach().cpu().numpy()

            # average accuracy if multi-dim
            assert 'accu' in results
            if results['accu'].ndim == 0:
                step_accu = math.nan if results['accu'] == math.nan else results['accu']
            else:
                assert results['accu'].ndim == 1
                empty = True
                for acc in results['accu']:
                    if not np.isnan(acc):
                        empty = False
                        break
                step_accu = math.nan if empty else np.nanmean(results['accu'])

            assert 'loss' in results
            progress_bar.set_description(
                '[%s] loss: %.5f, accu: %.5f'
                % (stage, results['loss'], step_accu)
            )

            if self.logger is not None:
                self.logger.add_scalar(
                    '%s/step/loss' % stage,
                    results['loss'],
                    self.step[stage]
                )
                self.logger.add_scalar(
                    '%s/step/accu' % stage,
                    -1 if math.isnan(step_accu) else step_accu,
                    self.step[stage]
                )

            result_list.append(results)

        summary = dict()
        if stage_config['revert']:
            reverter = Reverter(data_gen)
            result_collection_blacklist = reverter.revertible

            scores = dict()
            progress_bar = tqdm(
                reverter.on_batches(result_list),
                total=len(reverter.data_list),
                dynamic_ncols=True,
                ncols=get_tty_columns(),
                desc='[Data index]'
            )
            for reverted in progress_bar:
                data_idx = reverted['idx']
                scores[data_idx] = reverted['score']
                # info = '[%s] ' % data_idx
                # info += ', '.join(
                #     '%s: %.3f' % (key, val)
                #     for key, val in scores[data_idx].items()
                # )
                info = '[%s] mean score: %.3f' % (data_idx, np.mean(list(scores[data_idx].values())))
                progress_bar.set_description(info)

            # summerize score of each class over data indices
            cls_scores = {
                cls: np.mean([
                    scores[data_idx][cls] for data_idx in scores
                ])
                for cls in class_names
            }
            cls_scores.update({
                'mean': np.mean([
                    cls_scores[cls]
                    for cls in class_names
                ])
            })

            summary['scores'] = scores
            summary['cls_scores'] = cls_scores

        else:
            result_collection_blacklist = []

        # collect results except those revertible ones, e.g., accu, loss
        summary.update({
            key: np.nanmean(
                np.vstack([result[key] for result in result_list]),
                axis=0
            )
            for key in result_list[0].keys()
            if key not in result_collection_blacklist
        })

        # process 1D array accu to dictionary of each class score
        if len(summary['accu']) > 1:
            assert len(summary['accu']) == len(class_names)
            summary['cls_accu'] = {
                cls: summary['accu'][i]
                for (i, cls) in enumerate(class_names)
            }
            summary['accu'] = summary['accu'].mean()

        # print summary info
        print('Average: ' + ', '.join([
            '%s: %.3f' % (key, val)
            for (key, val) in  summary.items()
            if not isinstance(val, dict)
        ]))

        if 'cls_scores' in summary:
            print('Class score: ' + ', '.join([
                '%s: %.3f' % (key, val)
                for (key, val) in  summary['cls_scores'].items()
            ]))

        return summary

    def save(self, ckpt_dir):
        for key in self.handlers:
            self.handlers[key].save(os.path.join(ckpt_dir, key + '.pt'))

with open('./learning.json5') as f:
    config = json5.load(f)

# - GPUs
gpus = ",".join([str(idx) for idx in config['gpus']])
if len(gpus) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.backends.cudnn.enabled = True

trainer = Trainer(config, logger)

timer = time.time()
start = timer
init_epoch = 1
best = 0
try:
    for epoch in range(init_epoch, init_epoch + config['epochs']):
        epoch_info(epoch - 1, init_epoch + config['epochs'] - 1)

        for stage in config['stage']:
            if (epoch - init_epoch) % config['stage'][stage]['period'] != 0:
                continue

            summary = trainer.run(stage)

            # handle the case of 'scores' due to dual dictionary
            if 'scores' in summary:
                score = summary['cls_scores']['mean']
                file_path = os.path.join(
                    args.log_dir,
                    '%03d-%.5f.json' % (epoch, score)
                )
                with open(file_path, 'w') as f:
                    json.dump(summary.pop('scores'), f, indent=2)

                # XXX
                if stage == 'valid_target' and score > best:
                    best = score
                    trainer.save(args.ckpt_dir)

            for (key, val) in summary.items():
                if isinstance(val, dict):
                    logger.add_scalars('%s/epoch/%s' % (stage, key), val, epoch)
                else:
                    logger.add_scalar('%s/epoch/%s' % (stage, key), val, epoch)


except KeyboardInterrupt:
    logger.close()

print('Total:', time.time()-start)
print('Finished Training')
