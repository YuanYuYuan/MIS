# PDDCA Brainstem and Parotids Segmentation

Total 4 classes: background, brainstem, left/right parotids

Note that this script is for inference only.
Hence, no label information required. But if you have label of images
and want to evaluate the performance, or further training based on the model,
please see [here](../../exp/bs-ptd-nrrd).

## Latest results

Dice score of validation on PDDCA dataset (train/valid: 33/15)

| ROI     | Brainstem   | Left Parotid   | Right Parotid   | Average   |
| ------- | ------------ | -------------- | --------------- | --------- |
| Score   | 0.86830      | 0.75442        | 0.75059         | 0.79111   |


## How to use

### Prerequisites

Check the [installation](../../README.md) if needed.

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
    Brainstem: 1
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

The output results will be in NRRD format like

```bash
OUTPUT_DIR
├── DATA_INDEX
│   └── structures
│       ├── BrainStem.nrrd
│       ├── Parotid_L.nrrd
│       └── Parotid_R.nrrd
...
```
