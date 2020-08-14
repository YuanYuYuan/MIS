# Optic Nerves & Chiasm Segmentation with Bounding Boxes

This is a sample script for segmentation on optic nerves and chiasm
based on preprocessed given bounding boxes.

* This model focus on segmentation of optic nerves & chiasm,
and is supposed to be the 2nd stage in two-stage segmentation,
that is the data are given with corresponding bounding boxes.
* We improve the model's performance compared with the [previous version](https://github.com/YuanYuYuan/MIS/tree/master/deployments/visual-system) by
** restricting the ROI,
** more data preprocessing: affine + elastic augmentation, larger contrast range, etc.
** correct data sampling mechanism while training.


## Validation performance on PDDCA dataset

```yaml
Left Optic Nerve  : 0.6560
Right Optic Nerve : 0.6384
Chiasm            : 0.5244
Average           : 0.6063
```

## Usage

Click [here](https://yuanyuyuan.github.io/MIS/deployments/visual-system-bbox/) to see
the instruction.
