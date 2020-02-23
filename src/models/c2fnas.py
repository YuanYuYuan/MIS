import torch.nn as nn
import torch


class Operations(nn.Module):

    def __init__(
        self,
        mode: str,
        n_channels: int,
        merge: int = 0,
    ):
        super().__init__()

        if mode == '2D':
            core = nn.Conv3d(
                n_channels,
                n_channels,
                kernel_size=(3, 3, 1),
                padding=(1, 1, 0),
                bias=False
            )

        elif mode == '3D':
            core = nn.Conv3d(
                n_channels,
                n_channels,
                kernel_size=3,
                padding=1,
                bias=False
            )

        elif mode == 'P3D':
            core = nn.Sequential(
                nn.Conv3d(
                    n_channels,
                    n_channels,
                    kernel_size=(3, 3, 1),
                    padding=(1, 1, 0),
                    bias=False
                ),
                nn.Conv3d(
                    n_channels,
                    n_channels,
                    kernel_size=(1, 1, 3),
                    padding=(0, 0, 1),
                    bias=False
                )
            )
        else:
            raise NotImplementedError

        self.op = nn.Sequential(
            nn.ReLU(),
            core,
            nn.InstanceNorm3d(n_channels, affine=True)
        )

        self.preprocessing_op_1 = nn.Conv3d(
            n_channels,
            n_channels,
            kernel_size=1,
            bias=False
        )

        # single input
        if merge == 0:
            self.preprocessing_op_2 = None

        # additional input in the same layer
        elif merge == 1:
            self.preprocessing_op_2 = nn.Conv3d(
                n_channels,
                n_channels,
                kernel_size=1,
                bias=False
            )

        # additional input from lower layer
        elif merge == 2:
            self.preprocessing_op_2 = UpSample(n_channels * 2)

        self.postprocessing_op = nn.Conv3d(
            n_channels,
            n_channels,
            kernel_size=1,
            bias=False
        )

    def forward(self, x, y=None):
        output = self.preprocessing_op_1(x)
        if y is not None:
            output += self.preprocessing_op_2(y)
        output = self.op(output)
        output = self.postprocessing_op(output)
        return output


class DownSample(nn.Module):

    def __init__(
        self,
        n_channels: int,
    ):
        super().__init__()
        out_channels = n_channels * 2
        self.op = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(
                n_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm3d(out_channels, affine=True)
        )

    def forward(self, x):
        return self.op(x)


class UpSample(nn.Module):

    def __init__(
        self,
        n_channels: int,
    ):
        super().__init__()
        out_channels = n_channels // 2
        self.op = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear"),
            nn.ReLU(),
            nn.Conv3d(
                n_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.InstanceNorm3d(out_channels, affine=False)
        )

    def forward(self, x):
        return self.op(x)


class Encoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        n_filters: int = 32,
    ):
        super().__init__()
        depth = 5
        filters = [(n_filters * (2 ** i)) for i in range(depth)]
        self.input = nn.Sequential(
            nn.Conv3d(
                in_channels,
                filters[0],
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm3d(filters[0], affine=True),
            nn.ReLU(),
            nn.Conv3d(
                filters[0],
                filters[1],
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm3d(filters[1], affine=True),
        )

        self.downsamples = nn.ModuleList([
            DownSample(filters[i])
            for i in range(1, depth-1)
        ])
        self.layer_1 = nn.ModuleList([
            Operations(mode, filters[1], merge)
            for (mode, merge) in zip(['2D', 'P3D'], [0, 1])
        ])
        self.layer_2 = nn.ModuleList([
            Operations(mode, filters[2])
            for mode in ['3D', '3D']
        ])
        self.layer_3 = nn.ModuleList([
            Operations(mode, filters[3])
            for mode in ['3D']
        ])
        self.layer_4 = nn.ModuleList([
            Operations(mode, filters[4])
            for mode in ['2D']
        ])

    def forward(self, x):

        x = self.input(x)
        skips = []

        # layer 1
        shortcut = x
        x = self.layer_1[0](x)
        x = self.layer_1[1](x, shortcut)
        skips.append(x)

        # layer 2
        x = self.downsamples[0](x)
        x = self.layer_2[0](x)
        x = self.layer_2[1](x)
        skips.append(x)

        # layer 3
        x = self.downsamples[1](x)
        x = self.layer_3[0](x)
        skips.append(x)

        # layer 4
        x = self.downsamples[2](x)
        x = self.layer_4[0](x)

        return x, skips


class Decoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 512,
        n_filters: int = 32,
        n_classes: int = 3,
    ):
        super().__init__()
        depth = 5
        filters = [(n_filters * (2 ** i)) for i in range(depth)]

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(
                filters[1] * 2,
                filters[1],
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm3d(filters[1], affine=True),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0, mode='trilinear'),
            nn.Conv3d(
                filters[1],
                filters[0],
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm3d(filters[0], affine=True),
            nn.ReLU(),
            nn.Conv3d(
                filters[0],
                n_classes,
                kernel_size=1,
                bias=False
            )
        )

        self.layer_3 = nn.ModuleList([
            Operations(mode, filters[3], merge)
            for (mode, merge) in zip(['3D', '3D'], [2, 0])
        ])
        self.layer_2 = nn.ModuleList([
            Operations(mode, filters[2], merge)
            for (mode, merge) in zip(['P3D'], [2])
        ])
        self.layer_1 = nn.ModuleList([
            Operations(mode, filters[1], merge)
            for (mode, merge) in zip(['3D', '2D', '3D'], [2, 0, 1])
        ])

    def forward(self, x, skips):

        # layer 3
        x = self.layer_3[0](skips.pop(), x)
        x = self.layer_3[1](x)

        # layer 2
        x = self.layer_2[0](skips.pop(), x)

        # layer 1
        x = self.layer_1[0](skips.pop(), x)
        shortcut_1 = x
        x = self.layer_1[1](x)
        shortcut_2 = x
        x = self.layer_1[2](x, shortcut_1)
        x = self.output(torch.cat((shortcut_2, x), dim=1))

        return x


class C2FNASNet(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        n_filters: int = 32,
        n_classes: int = 3,
    ):
        super().__init__()
        depth = 5
        out_channels = n_filters * (2**(depth-1))
        self.encoder = Encoder(in_channels, n_filters)
        self.decoder = Decoder(out_channels, n_filters, n_classes)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        return x
