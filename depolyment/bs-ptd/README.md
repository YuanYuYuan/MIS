# PDDCA Brainstem and Parotids Segmentation

Total 4 classes: background, brainstem, left/right parotids

Note that this script is for inference only, no training and validation supported.
For further re-training, please see [here](../../exp).

This program doesn't require any label information by default.
If you have labels of the images and want to evaluate the performance after inference,
please see here.

## Latest results

Dice score of validation on PDDCA dataset (train/valid: 33/15)

| ROI   | Brain Stem | Left Parotid | Right Parotid | Average |
|-------|------------|--------------|---------------|---------|
| Score | 0.8782     | 0.7223       | 0.7404        | 0.7803  |


## How to use

### Download the trained model

```bash
make download_model
```

There will be a trained model weight downloaded and named as _model.pt_.

### Check the data format

The required data format is

```bash
DATA_DIR
├── DATA_INDEX
│   └──  img.nrrd
...
```

### Configure the inference behavior

_./infering.yaml_
```bash
gpus: 0                 <-- YOU CAN SET "gpus: 0,1" IF THERE'RE TWO AVAILABE GPUS
output_threshold: 0.2
output_dir: outputs
model_weight: model.pt

loader:
  name: NRRDLoader
  data_dir: data        <-- SPECIFY A DATA DIRECTORY
  roi_map:
    BrainStem: 1
    Parotid_L: 2
    Parotid_R: 3
  spacing: 1
  test: false
  resample: false

generator:
  BlockGenerator : ...

  Augmentor      : ...

  BatchGenerator :
    n_workers    : 1
    batch_size   : 12      <-- CHOOSE A PROPER BATCH SIZE
    verbose      : False

model: ...
```


### Make inference

```bash
make infer
```

The output results will be in NIfTI format like

```bash
OUTPUT_DIR
├── DATA_INDEX.nii.gz
...
```
