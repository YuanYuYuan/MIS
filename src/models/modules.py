from torch import nn
import torch
import math


class ConvBlock(nn.Module):

    def __init__(
        self,
        dim='3D',
        ch_in=16,
        ch_out: int = None,
        preprocess=False,
        postprocess=True,
    ):
        super().__init__()

        self.op = nn.Sequential()
        self.ch_in = ch_in
        self.ch_out = ch_in if ch_out is None else ch_out

        if preprocess:
            self.op.add_module(
                'preprocess_norm',
                nn.InstanceNorm3d(self.ch_in, affine=True)
            )
            self.op.add_module(
                'preprocess_acti',
                nn.ReLU(inplace=True)
            )

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
                    self.ch_in,
                    self.ch_out,
                    kernel_size=(1, 1, 3),
                    padding=(0, 0, 1),
                    bias=False
                )
            )

        if postprocess:
            self.op.add_module(
                'postprocess_norm',
                nn.InstanceNorm3d(self.ch_out, affine=True)
            )
            self.op.add_module(
                'postprocess_acti',
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.op(x)


class UpSample(nn.Module):

    def __init__(self, ch_in=16):
        super().__init__()
        ch_out = ch_in // 2
        self.op = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear"),
            nn.Conv3d(ch_in, ch_out, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.op(x)


class DownSample(nn.Module):

    def __init__(self, ch_in=16):
        super().__init__()
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
        return self.op(x)


class LatentReparametrization(nn.Module):

    def __init__(self, in_shape, z_dim=128):
        super().__init__()
        in_dim = math.prod(in_shape)
        self.op = nn.ModuleDict({
            'mean': nn.Linear(in_dim, z_dim),
            'std': nn.Linear(in_dim, z_dim),
        })
        if isinstance(in_shape, list):
            in_shape = tuple(in_shape)
        self.in_shape = in_shape

    def forward(self, x):
        assert x.shape[1:] == self.in_shape
        x = torch.flatten(x, start_dim=1)
        mean = self.op['mean'](x)
        std = self.op['std'](x)
        std = nn.functional.softplus(std) + 1e-6
        return mean, std


class LatentReconstruction(nn.Module):

    def __init__(self, out_shape, z_dim=128):
        super().__init__()
        self.op = nn.Sequential(
            nn.Linear(z_dim, math.prod(out_shape)),
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
