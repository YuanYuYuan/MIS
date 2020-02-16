#!/usr/bin/env python3
from torchsummary import summary
import argparse
import models
import yaml
import os


parser = argparse.ArgumentParser()
parser.add_argument(
    '--model-config',
    required=True,
)
parser.add_argument(
    '--gpus',
    default=None,
)
args = parser.parse_args()

if args.gpus is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

# load model template
with open(args.model_config) as f:
    model_config = yaml.safe_load(f)
model_name = model_config.pop('name')
model = getattr(models, model_name)(**model_config)

if args.gpus is not None:
    model = model.cuda()

input_size = (model_config['in_channels'],)
input_size += tuple(model_config['in_shape'])
summary(model, input_size=input_size)
