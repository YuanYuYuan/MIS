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
    '--config',
    default='./training.json5',
    help='training config',
)
parser.add_argument(
    '--log-dir',
    default='_logs',
    help='training logs',
)
parser.add_argument(
    '--ckpt-dir',
    default='_ckpts',
    help='training checkpoints',
)
parser.add_argument(
    '--pause-ckpt-dir',
    default='_pause_ckpts',
    help='pause checkpoints',
)
parser.add_argument(
    '--test',
    default=False,
    action='store_true',
    help='check feasibility',
)
args = parser.parse_args()

logger = SummaryWriter(args.log_dir)
os.makedirs(args.ckpt_dir, exist_ok=True)
os.makedirs(args.pause_ckpt_dir, exist_ok=True)


NO_TAG = 'no_tag'

class Trainer:

    def __init__(self, config, logger=None, test=False):

        self.logger = logger
        self.step = dict()
        self.config = config

        # task variables
        self.running_task = None
        self.tasks = config['task']
        for (task_name, task_config) in self.tasks.items():
            self.tasks[task_name]['need_backward'] = any(list(
                task_config['toggle'].values()
            ))

        # setup models and optimzers
        self.handlers = dict()
        self.optims = dict()
        self.lr_schedulers = dict()
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
            if 'lr_scheduler' in cfg:
                self.lr_schedulers[key] = getattr(
                    torch.optim.lr_scheduler,
                    cfg['lr_scheduler'].pop('name'),
                )(self.optims[key], **cfg['lr_scheduler'])

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
            if test:
                data_list = data_list[:1]
            loader_config = data_config['loader']
            loader_name = loader_config.pop('name')
            data_loader = DataLoader(loader_name, **loader_config)
            data_loader.set_data_list(data_list)
            self.generators[key] = DataGenerator(data_loader, cfg['struct'])


    def run_task(self, task_config, formatted_batch, need_revert=False):
        # formatted batch: {tag: {key: data}}

        # feed forward tag batch
        data = dict()
        for tag in formatted_batch:
            tag_data = formatted_batch[tag]
            if isinstance(task_config['forward'], list):
                modules_to_run = task_config['forward']
            else:
                if tag not in task_config['forward']:
                    continue
                modules_to_run = task_config['forward'][tag]

            with torch.set_grad_enabled(task_config['need_backward']):
                for mod in modules_to_run:
                    tag_data.update(self.handlers[mod].model(tag_data))

            if tag == NO_TAG:
                data.update(tag_data)
            else:
                data.update({'%s_%s' % (key, tag): tag_data[key] for key in tag_data})

        # evaluate the performance
        results = self.metrics[task_config['metric']](data)

        # backpropagation
        if task_config['need_backward']:
            results['loss'].backward()
            for (key, toggle) in task_config['toggle'].items():
                if toggle:
                    self.optims[key].step()
                    self.optims[key].zero_grad()
                    if key in self.lr_schedulers:
                        self.lr_schedulers[key].step()

        # compute match for dice score of each case after reversion
        if need_revert:
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

        return results



    def run(self, stage):
        stage_config = self.config['stage'][stage]

        # build data flow from the given data generator
        # single data flow
        if isinstance(stage_config['generator'], str):
            data_gen = self.generators[stage_config['generator']]
            class_names = data_gen.struct['DL'].ROIs
            n_steps = len(data_gen)
            gen_tags = None

        # multiple data flows
        elif isinstance(stage_config['generator'], dict):
            gens = [self.generators[cfg] for cfg in stage_config['generator'].values()]
            data_gen = zip(*gens)
            class_names = gens[0].struct['DL'].ROIs
            n_steps = min([len(g) for g in gens])
            gen_tags = list(stage_config['generator'].keys())

        else:
            raise TypeError('generator of type %s is not supported.' % type(stage_config['generator']))

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

        task_result_list = {task_name: [] for task_name in stage_config['task']}
        need_revert = 'revert' in stage_config and stage_config['revert']
        for batch in progress_bar:

            # format the batch into {tag: {key: data}}
            if gen_tags is None:
                assert isinstance(batch, dict)
                formatted_batch = {NO_TAG: dict()}
                for key in batch:
                    if torch.cuda.device_count() >= 1:
                        formatted_batch[NO_TAG][key] = batch[key].cuda()
                    else:
                        formatted_batch[NO_TAG][key] = batch[key]
            else:
                formatted_batch = dict()
                for (tag, tag_batch) in zip(gen_tags, batch):
                    tag_data = dict()
                    for key in tag_batch:
                        if torch.cuda.device_count() >= 1:
                            tag_data[key] = tag_batch[key].cuda()
                        else:
                            tag_data[key] = tag_batch[key]
                    formatted_batch[tag] = tag_data

            # execute each task in this stage
            for task_name in stage_config['task']:
                task_config = self.tasks[task_name]

                if 'period' in task_config \
                    and self.step[stage] % task_config['period'] != 0:
                    continue

                # modify the status of modules if the running task changed
                if self.running_task != task_name:

                    # toggle trainable parameters of each module
                    for (key, toggle) in task_config['toggle'].items():
                        self.handlers[key].model.train(toggle)
                        for param in self.handlers[key].model.parameters():
                            param.requires_grad = toggle

                    self.running_task = task_name

                task_result = self.run_task(
                    task_config,
                    formatted_batch,
                    need_revert=need_revert,
                )

                # detach all the results, move them to CPU, and convert them to numpy
                for key in task_result:
                    task_result[key] = task_result[key].detach().cpu().numpy()

                # average accuracy if multi-dim
                assert 'accu' in task_result
                if task_result['accu'].ndim == 0:
                    step_accu = math.nan if task_result['accu'] == math.nan else task_result['accu']
                else:
                    assert task_result['accu'].ndim == 1
                    empty = True
                    for acc in task_result['accu']:
                        if not np.isnan(acc):
                            empty = False
                            break
                    step_accu = math.nan if empty else np.nanmean(task_result['accu'])

                assert 'loss' in task_result
                progress_bar.set_description(
                    '[%s][%s] loss: %.5f, accu: %.5f'
                    % (stage, task_name, task_result['loss'], step_accu)
                )

                if self.logger is not None:
                    self.logger.add_scalar(
                        '%s/%s/step/loss' % (stage, task_name),
                        task_result['loss'],
                        self.step[stage]
                    )
                    self.logger.add_scalar(
                        '%s/%s/step/accu' % (stage, task_name),
                        -1 if math.isnan(step_accu) else step_accu,
                        self.step[stage]
                    )

                task_result_list[task_name].append(task_result)

            self.step[stage] += 1

        # summarize the  result list
        task_summary = dict()
        for (task_name, result_list) in task_result_list.items():

            if len(result_list) == 0:
                continue

            summary = dict()

            if need_revert:
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
                    info = '[%s] mean score: %.3f' % (data_idx, np.mean(list(scores[data_idx].values())))
                    progress_bar.set_description(info)

                # summerize score of each class over data indices
                cls_scores = {
                    cls: np.mean([
                        scores[data_idx][cls] for data_idx in scores
                    ])
                    for cls in class_names
                }

                summary['scores'] = scores
                summary['cls_scores'] = cls_scores
                summary['cls_mean'] = np.mean(list(cls_scores.values()))

            else:
                result_collection_blacklist = []

            # collect results except those revertible ones, e.g., collect accu, loss
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
                assert len(summary['accu']) == len(class_names), (len(summary['accu']), len(class_names))
                summary['cls_accu'] = {
                    cls: summary['accu'][i]
                    for (i, cls) in enumerate(class_names)
                }
                summary['accu'] = summary['accu'].mean()

            # print summary info
            print('[%s][%s] Average: ' % (stage, task_name) + ', '.join([
                '%s: %.3f' % (key, val)
                for (key, val) in  summary.items()
                if not isinstance(val, dict)
            ]))

            if 'cls_scores' in summary:
                print('Class score: ' + ', '.join([
                    '%s: %.3f' % (key, val)
                    for (key, val) in  summary['cls_scores'].items()
                ]))
                print('Class mean: %.3f' % summary['cls_mean'])

            task_summary[task_name] = summary

        return task_summary

    def save(self, ckpt_dir):
        for key in self.handlers:
            self.handlers[key].save(os.path.join(ckpt_dir, key + '.pt'))

with open(args.config) as f:
    config = json5.load(f)

# - GPUs
if 'gpus' in config:
    gpus = ",".join([str(idx) for idx in config['gpus']])
else:
    gpus = ""
if len(gpus) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.backends.cudnn.enabled = True

trainer = Trainer(config, logger, test=args.test)

timer = time.time()
start = timer
init_epoch = 1
terminated = False
best = 0
best_epoch = 0
n_stagnation = 0

try:
    for epoch in range(init_epoch, init_epoch + config['epochs']):

        if terminated:
            break

        print()
        epoch_info(epoch - 1, init_epoch + config['epochs'] - 1)

        for stage in config['stage']:
            if 'period' in config['stage'][stage] and (epoch - init_epoch) % config['stage'][stage]['period'] != 0:
                continue

            for (task, summary) in trainer.run(stage).items():

                # handle the case of 'scores' due to double dictionaries structure
                if 'scores' in summary:
                    score = summary['cls_mean']
                    file_path = os.path.join(
                        args.log_dir,
                        '%03d-%.5f.json' % (epoch, score)
                    )
                    with open(file_path, 'w') as f:
                        json.dump(summary.pop('scores'), f, indent=2)

                for (key, val) in summary.items():
                    if isinstance(val, dict):
                        logger.add_scalars('%s/%s/epoch/%s' % (stage, task, key), val, epoch)
                    else:
                        logger.add_scalar('%s/%s/epoch/%s' % (stage, task, key), val, epoch)

                # XXX should not check twice in different task
                if 'checkpoint' in config and stage == config['checkpoint']['stage']:
                    new_score = summary[config['checkpoint']['metric']]
                    if config['checkpoint']['mode'] == 'ascending':
                        improved  = new_score > best
                    elif config['checkpoint']['mode'] == 'descending':
                        improved  = new_score < best
                    else:
                        raise ValueError('The mode should be either ascending or descending.')

                    if improved:
                        print('Score improved from %.3f to %.3f.' % (best, new_score))
                        best = new_score
                        best_epoch = epoch
                        n_stagnation = 0
                        trainer.save(args.ckpt_dir)
                    else:
                        n_stagnation += 1
                        print('Best: %.3f at epoch %03d, stagnation: %d.' % (best, best_epoch, n_stagnation))

                    if n_stagnation > config['checkpoint']['early_stop']:
                        print('Early stopped.')
                        terminated = True
                        break

                    # if n_stagnation > config['checkpoint']['lr_decay'] and \
                    #     n_stagnation % config['checkpoint']['lr_decay'] == 1:
                    #     print('Learnging rate decayed.')


        for (key, opt) in trainer.optims.items():
            logger.add_scalar('lr_rate/%s' % key, opt.param_groups[0]['lr'], epoch)
        logger.add_scalar('best', best, epoch)

except KeyboardInterrupt:
    print('Paused the training.')

trainer.save(args.pause_ckpt_dir)
logger.close()
print('Total:', time.time()-start)
print('Finished Training')
