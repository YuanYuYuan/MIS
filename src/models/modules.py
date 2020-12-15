from torch import nn
import torch.nn.functional as F
import torch
import numpy as np


def acti(name):
    assert name in ('relu', 'leaky_relu')
    if name  == 'relu':
        return nn.ReLU(inplace=True)
    else:
        return nn.LeakyReLU(0.2, inplace=True)

class ConvBlock(nn.Module):

    def __init__(
        self,
        dim='3D',
        ch_in=16,
        ch_out: int = None,
        preprocess=False,
        postprocess=True,
        activation='relu',
        tag=None,
    ):
        super().__init__()

        self.op = nn.Sequential()
        self.ch_in = ch_in
        self.ch_out = ch_in if ch_out is None else ch_out
        self.dim = dim
        self.tag = tag

        if preprocess:
            self.op.add_module(
                'preprocess_norm',
                nn.InstanceNorm3d(self.ch_in, affine=True)
            )
            # FIXME
            self.op.add_module('preprocess_acti', acti(activation))

        if dim == '2D':
            self.op.add_module(
                '2D',
                nn.Conv3d(
                    self.ch_in,
                    self.ch_out,
                    kernel_size=(3, 3, 1),
                    padding=(1, 1, 0),
                    bias=False
                )
            )

        elif dim == '3D':
            self.op.add_module(
                '3D',
                nn.Conv3d(
                    self.ch_in,
                    self.ch_out,
                    kernel_size=3,
                    padding=1,
                    bias=False
                )
            )

        elif dim == 'P3D':
            self.op.add_module(
                'P3D_1',
                nn.Conv3d(
                    self.ch_in,
                    self.ch_out,
                    kernel_size=(3, 3, 1),
                    padding=(1, 1, 0),
                    bias=False
                )
            )
            self.op.add_module(
                'P3D_2',
                nn.Conv3d(
                    self.ch_out,
                    self.ch_out,
                    kernel_size=(1, 1, 3),
                    padding=(0, 0, 1),
                    bias=False
                )
            )

        elif dim == 'slice':
            self.op.add_module(
                'slice_1',
                nn.Conv3d(
                    in_channels=self.ch_in,
                    out_channels=self.ch_out,
                    kernel_size=(5, 5, 16),
                    padding=(2, 2, 0),
                ),
            )
            self.op.add_module(
                'slice_2',
                nn.Conv3d(
                    in_channels=self.ch_out,
                    out_channels=self.ch_out,
                    kernel_size=(3, 3, 1),
                    padding=(1, 1, 0),
                ),
            )

        if postprocess:
            self.op.add_module(
                'postprocess_norm',
                nn.InstanceNorm3d(self.ch_out, affine=True)
            )
            # FIXME
            self.op.add_module('postprocess_acti', acti(activation))

    def forward(self, x):
        assert x.shape[1] == self.ch_in, (self.dim, x.shape, self.ch_in, self.ch_out, self.tag)
        return self.op(x)


class UpSample(nn.Module):

    def __init__(self, ch_in=16, align_corners=True):
        super().__init__()
        self.ch_in = ch_in
        ch_out = ch_in // 2
        self.op = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode='trilinear',
                align_corners=align_corners
            ),
            nn.Conv3d(ch_in, ch_out, kernel_size=1, bias=False)
        )

    def forward(self, x):
        assert x.shape[1] == self.ch_in, (x.shape, self.ch_in)
        return self.op(x)


class DownSample(nn.Module):

    def __init__(self, ch_in=16):
        super().__init__()
        self.ch_in = ch_in
        ch_out = ch_in * 2
        self.op = nn.Conv3d(
            ch_in,
            ch_out,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

    def forward(self, x):
        assert x.shape[1] == self.ch_in, (x.shape, self.ch_in)
        return self.op(x)


class LatentReparametrization(nn.Module):

    def __init__(self, in_shape, z_dim=128):
        super().__init__()
        in_dim = np.prod(in_shape)
        self.op = nn.ModuleDict({
            'mean': nn.Linear(in_dim, z_dim),
            'std': nn.Linear(in_dim, z_dim),
        })
        if isinstance(in_shape, list):
            in_shape = tuple(in_shape)
        self.in_shape = in_shape

    def forward(self, x):
        assert x.shape[1:] == self.in_shape, (x.shape, self.in_shape)
        x = torch.flatten(x, start_dim=1)
        mean = self.op['mean'](x)
        std = self.op['std'](x)
        std = nn.functional.softplus(std) + 1e-6
        return mean, std


class LatentReconstruction(nn.Module):

    def __init__(self, out_shape, z_dim=128):
        super().__init__()
        self.op = nn.Sequential(
            nn.Linear(z_dim, np.prod(out_shape)),
            nn.ReLU(inplace=True),
        )
        if isinstance(out_shape, list):
            out_shape = tuple(out_shape)
        self.shape = (-1,) + out_shape

    def forward(self, x):
        assert len(x) == 2
        mean, std = x

        # sample by normal distribution
        if self.training:
            x = mean + std * torch.rand_like(std)
        else:
            x = mean

        x = self.op(x)
        x = torch.reshape(x, self.shape)
        return x

class Classifier(nn.Module):

    def __init__(
        self,
        in_shape=(32, 6, 6, 6),
        n_classes=2,
        activation='leaky_relu',
    ):
        super().__init__()
        in_dim = np.prod(in_shape)
        self.op = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_dim, 4096),
            acti(activation),

            nn.Dropout(),
            nn.Linear(4096, 1024),
            acti(activation),

            nn.Dropout(),
            nn.Linear(1024, n_classes),
        )
        if isinstance(in_shape, list):
            in_shape = tuple(in_shape)
        self.in_shape = in_shape

    def forward(self, x):
        assert x.shape[1:] == self.in_shape, (x.shape, self.in_shape)
        x = torch.flatten(x, start_dim=1)
        return self.op(x)


class IntoProb(nn.Module):
    def __init__(self, n_classes=6, add_noise=False):
        super().__init__()
        self.n_classes = n_classes

        # TODO
        self.add_noise = add_noise

    def forward(self, x):
        raw_shape = x.shape

        # prediction
        if len(raw_shape) == 5:
            assert raw_shape[1] == self.n_classes
            return F.softmax(x, dim=1)

        # ground truth
        else:
            assert len(raw_shape) == 4
            x = F.one_hot(x, self.n_classes).permute((0, 4, 1, 2, 3)).float()

        return x


class FCClassifier(nn.Module):
    def __init__(
        self,
        ch_in=256,
        n_classes=1,
    ):
        super().__init__()
        self.op = nn.Linear(ch_in, n_classes)

    def forward(self, x):
        # Global Average Pooling
        x = x.mean(dim=(2, 3, 4))
        x = self.op(x)
        return x
