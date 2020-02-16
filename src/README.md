# Medical Image Data Pipeline

This project is designed for the data feeding for deep learning models based on PyTorch.
Still under development, use it with caution.

## Installation

The used python version is 3.7.  And it's recommended to install locally.

```bash
python setup.py install --user
```

## Prepare data

In the following examples, we use the [PDDCA](http://www.imagenglab.com/newsite/pddca/) dataset, and name it as _data_.

## Examples

### Test model

```bash
./print_model.py --model-config configs/model.yaml            # For CPU only
./print_model.py --model-config configs/model.yaml --gpus 0   # For GPU
```

Sample output

```txt
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1       [-1, 32, 64, 64, 30]             896
       BatchNorm3d-2       [-1, 32, 64, 64, 30]              64
         InputConv-3       [-1, 32, 64, 64, 30]               0
            Conv3d-4       [-1, 64, 32, 32, 15]          16,448
       BatchNorm3d-5       [-1, 64, 32, 32, 15]             128
              ReLU-6       [-1, 64, 32, 32, 15]               0
            Conv3d-7       [-1, 64, 32, 32, 15]         110,656
       BatchNorm3d-8       [-1, 64, 32, 32, 15]             128
              ReLU-9       [-1, 64, 32, 32, 15]               0
           Conv3d-10       [-1, 64, 32, 32, 15]         110,656
        ConvBlock-11       [-1, 64, 32, 32, 15]               0
           Conv3d-12       [-1, 128, 16, 16, 7]          65,664
      BatchNorm3d-13       [-1, 128, 16, 16, 7]             256
             ReLU-14       [-1, 128, 16, 16, 7]               0
           Conv3d-15       [-1, 128, 16, 16, 7]         442,496
      BatchNorm3d-16       [-1, 128, 16, 16, 7]             256
             ReLU-17       [-1, 128, 16, 16, 7]               0
           Conv3d-18       [-1, 128, 16, 16, 7]         442,496
        ConvBlock-19       [-1, 128, 16, 16, 7]               0
           Conv3d-20         [-1, 256, 8, 8, 3]         262,400
      BatchNorm3d-21         [-1, 256, 8, 8, 3]             512
             ReLU-22         [-1, 256, 8, 8, 3]               0
           Conv3d-23         [-1, 256, 8, 8, 3]       1,769,728
      BatchNorm3d-24         [-1, 256, 8, 8, 3]             512
             ReLU-25         [-1, 256, 8, 8, 3]               0
           Conv3d-26         [-1, 256, 8, 8, 3]       1,769,728
        ConvBlock-27         [-1, 256, 8, 8, 3]               0
  ConvTranspose3d-28       [-1, 128, 16, 16, 6]         262,272
      BatchNorm3d-29       [-1, 128, 16, 16, 7]             256
             ReLU-30       [-1, 128, 16, 16, 7]               0
           Conv3d-31       [-1, 128, 16, 16, 7]         442,496
      BatchNorm3d-32       [-1, 128, 16, 16, 7]             256
             ReLU-33       [-1, 128, 16, 16, 7]               0
           Conv3d-34       [-1, 128, 16, 16, 7]         442,496
        ConvBlock-35       [-1, 128, 16, 16, 7]               0
  ConvTranspose3d-36       [-1, 64, 32, 32, 14]          65,600
      BatchNorm3d-37       [-1, 64, 32, 32, 15]             128
             ReLU-38       [-1, 64, 32, 32, 15]               0
           Conv3d-39       [-1, 64, 32, 32, 15]         110,656
      BatchNorm3d-40       [-1, 64, 32, 32, 15]             128
             ReLU-41       [-1, 64, 32, 32, 15]               0
           Conv3d-42       [-1, 64, 32, 32, 15]         110,656
        ConvBlock-43       [-1, 64, 32, 32, 15]               0
  ConvTranspose3d-44       [-1, 32, 64, 64, 30]          16,416
      BatchNorm3d-45       [-1, 32, 64, 64, 30]              64
             ReLU-46       [-1, 32, 64, 64, 30]               0
           Conv3d-47       [-1, 32, 64, 64, 30]          27,680
      BatchNorm3d-48       [-1, 32, 64, 64, 30]              64
             ReLU-49       [-1, 32, 64, 64, 30]               0
           Conv3d-50       [-1, 32, 64, 64, 30]          27,680
        ConvBlock-51       [-1, 32, 64, 64, 30]               0
           Conv3d-52        [-1, 4, 64, 64, 30]           3,460
       OutputConv-53        [-1, 4, 64, 64, 30]               0
================================================================
Total params: 6,503,332
Trainable params: 6,503,332
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.47
Forward/backward pass size (MB): 487.75
Params size (MB): 24.81
Estimated Total Size (MB): 513.03
----------------------------------------------------------------
```

### Train

```bash
./sample_generator.py \
    --loader-config configs/loader.yaml \
    --generator-config configs/generator.yaml
```

The output files are stored in 3D NIfTI (nii.gz) in the _outputs_folder.
One may view these images by [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php).

### Generate data list for training

```bash
./generate_data_list.py --loader-config configs/loader.yaml
```

Sample output: _data\_list.yaml_

```yaml
amount:
  test: 0
  total: 3
  train: 2
  valid: 1
list:
  test: []
  train:
  - 0522c0001
  - 0522c0003
  valid:
  - 0522c0002
loader:
- ROIs:
  - BrainStem
  - Parotid_L
  - Parotid_R
  data_dir: data
  data_list: test
  name: PDDCAParser
  preprocess_image: true
```
