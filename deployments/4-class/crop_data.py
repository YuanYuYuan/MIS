#!/usr/bin/env python3

import json
import numpy as np
import nibabel as nib
import os
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data-dir',
    default='./nifti',
    help='Specify the directory containing NIfTI images'
)
parser.add_argument(
    '--output-dir',
    default='./cropped',
    help='Specify the output directory'
)
parser.add_argument(
    '--crop-size',
    default=(196, 196, 196),
    type=int,
    nargs='+',
    help='Specify crop size, use the crop size in bbox.json if (0, 0, 0) given.'
)
args = parser.parse_args()

with open('./bbox.json') as f:
    bbox = json.load(f)

assert isinstance(args.crop_size, (list, tuple))
assert len(args.crop_size) == 3
crop_size = tuple(c for c in args.crop_size)
if crop_size == (0, 0, 0):
    print('Use the crop_size according to each bbox.')
else:
    print('Use the crop_size:', crop_size)

def crop(data, center, shape, dtype='image'):
    assert dtype in ['image', 'label']
    assert len(center) == len(shape)
    crop_idx = {'left': [], 'right': []}
    padding = {'left': [], 'right': []}

    for i in range(len(center)):
        left_corner = center[i] - (shape[i] // 2)
        right_corner = left_corner + shape[i]

        crop_idx['left'].append(max(0, left_corner))
        padding['left'].append(max(0, -left_corner))

        crop_idx['right'].append(min(right_corner, data.shape[i]))
        padding['right'].append(max(0, right_corner - data.shape[i]))

    crop_range = tuple(
        slice(lc, rc) for (lc, rc) in
        zip(crop_idx['left'], crop_idx['right'])
    )

    if len(crop_range) < 3:
        crop_range += (slice(None),) * (3 - len(crop_range))

    need_padding = False
    for key in padding:
        if sum(padding[key]) > 0:
            need_padding = True
            break

    if need_padding:
        zeros_shape = shape
        if dtype == 'image':
            output = np.ones(zeros_shape) * -1024
        else:
            output = np.zeros(zeros_shape)
        output[tuple(
            slice(lp, s - rp) for (lp, rp, s)
            in zip(padding['left'], padding['right'], shape)
        )] = data[crop_range]

        return output

    else:
        return data[crop_range]


os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'labels'), exist_ok=True)

def process(idx):
    # image
    image = np.asarray(nib.load(os.path.join(
        args.data_dir,
        'images',
        idx + '.nii.gz',
    )).dataobj)
    image = crop(image, bbox[idx]['center'], crop_size, dtype='image')
    nib.save(
        nib.Nifti1Image(image, affine=np.eye(4)),
        os.path.join(args.output_dir, 'images', idx + '.nii.gz')
    )

    # label
    label = np.asarray(nib.load(os.path.join(
        args.data_dir,
        'labels',
        idx + '.nii.gz',
    )).dataobj)
    label = crop(label, bbox[idx]['center'], crop_size, dtype='label')
    nib.save(
        nib.Nifti1Image(label, affine=np.eye(4)),
        os.path.join(args.output_dir, 'labels', idx + '.nii.gz')
    )


data_list = [
    f.split('/')[-1].split('.')[0]
    for f in glob(os.path.join(args.data_dir, 'images', '*.nii.gz'))
]


with Pool(os.cpu_count()) as pool:
    list(tqdm(
        pool.imap(process, data_list),
        total=len(data_list),
    ))
