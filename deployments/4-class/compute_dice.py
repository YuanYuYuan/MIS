#!/usr/bin/env python3

import nibabel as nib
import os
from glob import glob
import numpy as np
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--ground-truth',
    default='./nifti/labels',
    help='The directory containing the ground truth.'
)
parser.add_argument(
    '--prediction',
    default='./infer/outputs',
    help='The directory containing the model\' prediction.'
)
args = parser.parse_args()


def compute_dice(bin_x, bin_y):
    return 2 * np.sum(bin_x * bin_y) / np.sum(bin_x + bin_y)

def compute_score(x, y):
    return [
        compute_dice(
            np.array(x == target).astype(np.int),
            np.array(y == target).astype(np.int),
        )
        for target in range(1, 5)
    ]

def process(idx):
    return compute_score(
        np.asarray(nib.load(os.path.join(args.ground_truth, idx + '.nii.gz')).dataobj),
        np.asarray(nib.load(os.path.join(args.prediction, idx + '.nii.gz')).dataobj),
    )


gt_list = [
    f.split('/')[-1].split('.')[0]
    for f in glob(os.path.join(args.ground_truth, '*.nii.gz'))
]
assert len(gt_list) > 0, 'Empty ground truth list!'
pd_list = [
    f.split('/')[-1].split('.')[0]
    for f in glob(os.path.join(args.prediction, '*.nii.gz'))
]
assert len(pd_list) > 0, 'Empty prediction list!'

# scores: [n_cases x n_classes]
with Pool(os.cpu_count()) as pool:
    scores = np.asarray(list(pool.imap(process, pd_list)))

avg = scores.mean()
cls_avg = scores.mean(axis=0)
for (i, soi) in enumerate([
    'Brainstem',
    'Optic Chiasm',
    'Left Optic Nerve',
    'Right Optic Nerve',
]):
    print('%s: %.3f' % (soi, cls_avg[i]))
print('Average: %.3f' % avg)
print('\nCase average:', scores.mean(axis=1))
