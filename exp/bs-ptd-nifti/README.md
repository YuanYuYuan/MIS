# PDDCA Brainstem and Parotids Segmentation

Total 4 classes: background, brainstem, left/right parotids

## Recent results

Dice score of validation:

| ROI   | Brain Stem | Left Parotid | Right Parotid | Average |
|-------|------------|--------------|---------------|---------|
| Score | 0.8253     | 0.6351       | 0.7398        | 0.7334  |

Model: [[download]](https://drive.google.com/open?id=129p-xlP7S8Lf1v-KfyWuWhyUUtb5FT4i)

```bash
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1e4hROvGBmu1iGvaQaURliemB_avkdnt2' -O inference-graphs.tar.gz
```

## Preparation

### Case 1: To reproduce on PDDCA

1. Download the model to here, which should be named as "236-0.73342.pt"
2. Modify the `DATA_DIR` in the `loader` part in *data.yaml* for your PDDCA dataset.
See [here](../sample/README.md) for more details.

    *data.yaml*
    ```bash
    loader:
    - name: PDDCAParser
      data_dir: DATA_DIR # <-- MODIFY HERE
      data_list:
      ROIs:
      - BrainStem
      - Parotid_L
      - Parotid_R
      preprocess_image: true
                ⋮
    ```

### Case 2: Apply on another dataset

1. Download the model to here, which should be named as "236-0.73342.pt"
2. Copy and modify the *loader.yaml* for your custom dataset.
See [here](../sample/README.md) for more details.

    *loader.yaml*
    ```bash
    - name: PDDCAParser
      data_dir: DATA_DIR # <-- MODIFY HERE
      data_list: null
      ROIs:
      # - Mandible         # 1
      - BrainStem        # 2
      - Parotid_L        # 3
      - Parotid_R        # 4
      # - Submandibular_L  # 5
      # - Submandibular_R  # 6
      # - OpticNerve_L     # 7
      # - OpticNerve_R     # 8
      # - Chiasm           # 9
      preprocess_image: true
    ```
3. After setup the data loader, use *loader.yaml* to generate data list *data.yaml*.
Alos, see [here](../sample/README.md) for more details.

    ```bash
    make gen_data
    ```

## Directly predict

```bash
make predict CKPT="236-0.73342.pt"
```

The following table shows the result on PDDCA dataset.

*_predi_logs/prediction_score.txt*
```bash
+-----------+-----------+-----------+-----------+---------+
|     ID    | BrainStem | Parotid_L | Parotid_R |   AVG   |
+-----------+-----------+-----------+-----------+---------+
| 0522c0001 |  0.83343  |  0.80543  |  0.48085  | 0.70657 |
| 0522c0009 |  0.83740  |  0.79323  |  0.80995  | 0.81353 |
| 0522c0013 |  0.84756  |  0.49595  |  0.67225  | 0.67192 |
| 0522c0125 |  0.77753  |  0.52660  |  0.57364  | 0.62592 |
| 0522c0132 |  0.84347  |  0.76899  |  0.79858  | 0.80368 |
| 0522c0147 |  0.73418  |  0.61196  |  0.73624  | 0.69413 |
| 0522c0159 |  0.80828  |  0.69008  |  0.82127  | 0.77321 |
| 0522c0248 |  0.81264  |  0.77598  |  0.72246  | 0.77036 |
| 0522c0251 |  0.81601  |  0.58652  |  0.68940  | 0.69731 |
| 0522c0427 |  0.81712  |  0.56643  |  0.72456  | 0.70270 |
| 0522c0433 |  0.86394  |  0.65024  |  0.78221  | 0.76547 |
| 0522c0457 |  0.87440  |  0.50721  |  0.85031  | 0.74397 |
| 0522c0598 |  0.88552  |  0.81739  |  0.87950  | 0.86081 |
| 0522c0727 |  0.86333  |  0.50784  |  0.75955  | 0.71024 |
| 0522c0878 |  0.76484  |  0.42291  |  0.79688  | 0.66154 |
|    AVG    |  0.82531  |  0.63512  |  0.73984  | 0.73342 |
|    STD    |  0.04071  |  0.12770  |  0.10149  | 0.08997 |
+-----------+-----------+-----------+-----------+---------+
```

### Results

Below is a comparison between true label *_preprocessed_data/labels/0522c0009.nii.gz* and the prediction
*_predictions/0522c0009.nii.gz* on the preprocessed CT image *_preprocessed_data/images/0522c0009.nii.gz*.

![prediction](./pic/prediction.png)
<p align="center">
    Comparison on ITK-SNAP
</p>


## Continue Training

```bash
make retrain CKPT="236-0.73342.pt"
```

### Observe the performance

Run the following command and open http://localhost:6006 in browser.

```bash
make log
```

### Directly predict

make predict CKPT="236-0.73342.pt"
```
For more details, please see [here](../sample/README.md#Start-Training)
