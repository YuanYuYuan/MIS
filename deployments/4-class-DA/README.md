# Study of Domain Adaptation crossing datasets on 4 OARs CT Segmentation

## Prepare dataset

Please follow the [instruction](https://github.com/YuanYuYuan/MIS/tree/master/deployments/4-class#prepare-dataset)
and prepare source and target datasets (in NIfTI).

Next link the datasets under the _data_ folder.

```bash
ln -s SOURCE_DATASET data/source
ln -s TARGET_DATASET data/target
```
The file hierarchy of _data_ should look like below.

```bash
data
├── source
│   ├── images
│   │   ├── SAMPLE_1.nii.gz
│   │   ├── SAMPLE_2.nii.gz
│   │   ├── ...
│   └── labels
│       ├── SAMPLE_1.nii.gz
│       ├── SAMPLE_2.nii.gz
│       ├── ...
│
└── target
    ├── images
    │   ├── SAMPLE_1.nii.gz
    │   ├── SAMPLE_2.nii.gz
    │   ├── ...
    └── labels
        ├── SAMPLE_1.nii.gz
        ├── SAMPLE_2.nii.gz
        ├── ...
```

Specify the partition of data for the domain adaptation.
We need to setup three data lists

_./data_lists/source_train.json5_, which is used for supervised training on segmentation, domain adaptation as source

```json5
{
  "amount": AMOUNT,  // optional
  "list": [          // fill in the list of source samples for training
    SAMPLE_1,
    SAMPLE_2,
    ...
  ],
  "loader": { ... }
}

```

_./data_lists/valid_train.json5_, which is used to determine a best model for segmentation and domain adaptation


```json5
{
  "amount": AMOUNT,  // optional
  "list": [          // fill in the list of source samples for validation
    SAMPLE_1,
    SAMPLE_2,
    ...
  ],
  "loader": { ... }
}
```

_./data_lists/target.json5_, which is used for domain adaptation as target

```json5
{
  "amount": AMOUNT,  // optional
  "list": [          // fill in the list of target samples for domain adaptation
    SAMPLE_1,
    SAMPLE_2,
    ...
  ],
  "loader": { ... }
}
```


## Run experiements

### Domain Adaptation without DA

```bash
cd exps/without_DA
make test         // check the program runs normally first
make train
```

### Domain Adaptation with DA

```bash
cd exps/with_DA
make test         // check the program runs normally first
make train
```

### Inference and testing

The model will automatically save the best checkpoint of model based on the
performance of source validation set.
We can run the following commands in each experiment directory to produce the segmentation.

```bash
cd exps/with_DA  # or cd exps/without_DA
make infer
```

### Evaluate the performance

Please refer to [here](https://github.com/YuanYuYuan/MIS/tree/master/deployments/4-class#evaluate-the-performance).
