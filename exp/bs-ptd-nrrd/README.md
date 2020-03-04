# PDDCA Brainstem and Parotids Segmentation

Total 4 classes: background, brainstem, left/right parotids

## Configure data pipeline

### For training/validation

_data_list.yaml_
```bash
amount:
  test: 0
  total: 48
  train: 33
  valid: 15
list:
  test: []
  train: ...
  valid: ...
loader:
  name: NRRDLoader
  data_dir: data     <-- SPECIFY A DATA DIRECTORY
  roi_map:          <-- SPECIFY THE TARGETS
    Brainstem: 1
    Parotid_L: 2
    Parotid_R: 3
  spacing: 1
  test: false
  resample: false
```

### For inference

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

## How to use

### Train

```bash
make train
```

### Validate only

```bash
make validate CKPT=MODEL_CHECKPOINT
```

### Continue Training

```bash
make retrain CKPT=MODEL_CHECKPOINT
```

### Make inference

```bash
make infer
```

### View the validation score

```bash
cat _logs/scores.csv
```

### Monitor the training behavior

Run the following command and open http://localhost:6006 in browser.

```bash
make log
```
