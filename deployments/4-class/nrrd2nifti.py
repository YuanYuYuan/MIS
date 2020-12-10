#!/usr/bin/env python3

from glob import glob
import os
import numpy as np
import nrrd
from scipy import ndimage
import nibabel as nib
from multiprocessing import Pool
from tqdm import tqdm
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data-dir',
    default='./data',
    help='Specify the directory containing NRRD images'
)
parser.add_argument(
    '--output-dir',
    default='./nifti',
    help='Specify the output directory'
)
args = parser.parse_args()


data_list = [
    fn.split('/')[-2] for fn in
    glob(os.path.join(args.data_dir, '*/'))
]
assert len(data_list) > 0

IMAGE_TYPE = np.int16
LABEL_TYPE = np.uint8
SPACING = 1.2

# map of structure of interest (SOI)
SOI_MAP = {
    'Brainstem': 1,
    'Chiasm': 2,
    'OpticNerve_L': 3,
    'OpticNerve_R': 4,
}

os.makedirs(args.output_dir, exist_ok=True)
for key in ['images', 'labels']:
    os.makedirs(os.path.join(args.output_dir, key), exist_ok=True)

def get_spacing(nrrd_data):
    return tuple(np.diag(nrrd_data[1]['space directions']))

def load_image(data_idx):
    nrrd_data = nrrd.read(os.path.join(args.data_dir, data_idx, 'img.nrrd'))
    image = nrrd_data[0]
    spacing = get_spacing(nrrd_data)
    return image.astype(IMAGE_TYPE), spacing

def correct_orientation(data):
    return data[::-1, ::-1, ...]

def load_label(data_idx):
    label: np.Array = None
    spacing = None

    for (soi, value) in SOI_MAP.items():
        label_file = os.path.join(
            args.data_dir,
            data_idx,
            'structures',
            soi + '.nrrd',
        )
        if not os.path.exists(label_file):
            continue
        nrrd_data = nrrd.read(label_file)
        if label is None:
            label = nrrd_data[0]
            spacing = get_spacing(nrrd_data)
        else:
            label = np.maximum(label, nrrd_data[0] * value)
            assert spacing == get_spacing(nrrd_data)

    return label.astype(LABEL_TYPE), spacing

def convert(data_idx, order=2):
    image, spacing_1 = load_image(data_idx)
    label, spacing_2 = load_label(data_idx)
    assert spacing_1 == spacing_2
    assert image.shape == label.shape
    zoom = tuple(s / SPACING for s in spacing_1)

    # image
    image = ndimage.zoom(image, zoom, order=order, mode='nearest')
    image = image.astype(IMAGE_TYPE)
    image = correct_orientation(image)
    nib.save(
        nib.Nifti1Image(image, affine=np.eye(4)),
        os.path.join(args.output_dir, 'images', data_idx + '.nii.gz')
    )

    # label
    label = ndimage.zoom(label, zoom, order=0, mode='nearest')
    label = label.astype(LABEL_TYPE)
    label = correct_orientation(label)
    nib.save(
        nib.Nifti1Image(label, affine=np.eye(4)),
        os.path.join(args.output_dir, 'labels', data_idx + '.nii.gz')
    )

data_list = data_list[:2]
with Pool(os.cpu_count()) as pool:
    list(tqdm(pool.imap(convert, data_list), total=len(data_list)))


with open(os.path.join(args.output_dir, 'info.json'), 'w') as f:
    json.dump({'list': data_list, 'soi_map': SOI_MAP}, f, indent=2)

print('Ouputs have been stored in %s.' % args.output_dir)
