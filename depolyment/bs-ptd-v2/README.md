# Brainstem & Parotids Segmentation

## Prepare dataset

Place your dataset folder in here and name it as __data__. Or you may
directly create a link to the folder like this.

```bash
ln -s YOUR_DATASET data
```

## Download the trained model


## Usage

### Training

```bash
cd training
make retrain CKPT=MODEL_CHECKPOINT
```

### Evaluation

```bash
cd evaluation
make evaluate CKPT=MODEL_CHECKPOINT
```
