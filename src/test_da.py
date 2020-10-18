#!/usr/bin/env python3
import json5
from training import ModelHandler, Optimizer
from flows import MetricFlow, ModuleFlow
from tqdm import tqdm
from utils import get_tty_columns
from metrics import match_up
import torch
import math
import numpy as np


class Runner:

    def __init__(self, config, logger=None):
        self.logger = logger
        self.step = dict()

        with open(config) as f:
            self.config = json5.load(f)

        # setup models and optimzers
        self.handlers = dict()
        self.optims = dict()
        for (key, module_config) in self.config['module'].items():
            if 'ckpt' in module_config:
                ckpt = module_config['ckpt']
            else:
                ckpt = None
            self.handlers[key] = ModelHandler(
                module_config['config'],
                checkpoint=ckpt,
            )
            self.optims[key] = Optimizer(module_config['optim'])(self.handlers[key].model)

        self.metrics = {
            key: MetricFlow(config) for (key, config)
            in self.config['metric'].items()
        }

    def run(self, data_gen, stage, compute_match=False):
        stage_config = self.config['stage'][stage]

        n_steps = len(data_gen)
        progress_bar = tqdm(
            data_gen,
            total=n_steps,
            ncols=get_tty_columns(),
            dynamic_ncols=True,
            desc='[%s] Loss: %.5f, Accu: %.5f'
            % (stage, 0.0, 0.0)
        )

        if stage not in self.step:
            self.step[stage] = 1

        result_list = []
        for batch in progress_bar:

            self.step[stage] += 1

            # prepare data, merge batch from multiple generator if needed
            data = dict()
            if isinstance(batch, tuple):
                for key in batch[0]:
                    data[key] = torch.cat(sub_batch[key] for sub_batch in batch).cuda()
            else:
                assert isinstance(batch, dict)
                for key in batch:
                    data[key] = batch[key].cuda()

            # toggle trainable parameters of each module
            for key, toggle in stage_config['toggle'].items():
                self.handlers[key].model.train(toggle)

            # feed in data and run
            for key in stage_config['forward']:
                data.update(self.handlers.model[key](data))

            # compute loss and accuracy
            results = self.metrics[stage_config['metric']](data)

            # compute match for dice score of each case
            if compute_match:
                assert 'prediction' in data
                assert 'labels' in data
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

            if 'accu' in results:
                step_accu = np.nanmean(results['accu'])
            else:
                step_accu = math.nan

            if 'loss' in results:
                progress_bar.set_description(
                    '[%s] Loss: %.5f, Avg accu: %.5f'
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

        return result_list


runner = Runner('./learning.json5')


for epoch in range(init_epoch, init_epoch + config['epochs']):
