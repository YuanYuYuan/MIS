import torch.nn as nn

CONV_LAYER = {
    2: nn.Conv2d,
    3: nn.Conv3d
}

CONV_TRANSPOSE_LAYER = {
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d
}


class Normalization:

    def __init__(self, n_dim=3, norm_type='BatchNorm'):
        self.norm_type = norm_type
        if norm_type == 'GroupNorm':
            self._norm = getattr(nn, norm_type)
        elif norm_type in ['BatchNorm', 'InstanceNorm']:
            self._norm = getattr(nn, norm_type + str(n_dim) + 'd')

    def __call__(self, n_channels):
        if self.norm_type == 'GroupNorm':
            return self._norm(num_groups=8, num_channels=n_channels)
        else:
            return self._norm(n_channels)
