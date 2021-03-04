#!/usr/bin/env python3

import nibabel as nib
import os
from glob import glob
import numpy as np
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data-dir',
    default='./nifti',
    help='Specify the directory containing NIfTI images'
)
args = parser.parse_args()

file_list = glob(os.path.join(
    args.data_dir,
    './labels/*.nii.gz',
))
assert len(file_list) > 0, \
    "No nii.gz files found in %s/labels." % args.data_dir

def find_box(target, padding=20):
    assert isinstance(padding, int)
    assert len(target.shape) == 3

    indices = np.where(target)
    corner_1 = [
        int(max(min(idx) - padding, 0))
        for idx in indices
    ]
    corner_2 = [
        int(min(max(idx) + padding, margin))
        for (idx, margin) in zip(indices, target.shape)
    ]
    return {
        'center': [(l + r) // 2 for (l, r) in zip(corner_1, corner_2)],
        'shape': [(r - l) for (l, r) in zip(corner_1, corner_2)],
    }


file_name = lambda f: f.split('/')[-1].split('.')[0]
boxes = {
    file_name(f): find_box(np.asarray(nib.load(f).dataobj) > 0)
    for f in file_list
}

box_size = np.array([box['shape'] for box in boxes.values()])
print('Box size')
for (axis, avg, std) in zip(
    ['x', 'y', 'z'],
    box_size.mean(axis=0),
    box_size.std(axis=0),
):
    print('\t%s: %.2f ± %.2f' % (axis, avg, std))


print()
with open('./bbox.json', 'w') as f:
    json.dump(boxes, f, indent=2)

print('The list of bounding boxes have been stored into ./bbox.json.')
