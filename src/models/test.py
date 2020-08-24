import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv3d, Upsample, InstanceNorm3d
from torch.nn import Sigmoid, Softmax

class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, mode="3d"):
        super(ConvBlock, self).__init__()

        if mode not in ["2d", "2D", "3d", "3D", "p3d", "P3D"]:
            raise ValueError("Unknow mode for convolution")
        #endif

        pad_size = (kernel_size - 1) // 2

        if mode in ["2d", "2D"]:
            self.conv1 = Conv3d(
                inc,
                outc,
                kernel_size=(kernel_size, kernel_size, 1),
                stride=1,
                padding=(pad_size, pad_size, 0),
                bias=False
            )
            self.conv = self.conv2d
        #endif

        if mode in ["3d", "3D"]:
            self.conv1 = Conv3d(
                inc,
                outc,
                kernel_size=kernel_size,
                stride=1,
                padding=pad_size,
                bias=False
            )
            self.conv = self.conv3d
        #endif

        if mode in ["p3d", "P3D"]:
            self.conv1 = Conv3d(
                inc,
                outc,
                kernel_size=(kernel_size, kernel_size, 1),
                stride=1,
                padding=(pad_size, pad_size, 0),
                bias=False
            )
            self.conv2 = Conv3d(
                outc,
                outc,
                kernel_size=(1, 1, kernel_size),
                stride=1,
                padding=(0, 0, pad_size),
                bias=False
            )
            self.conv = self.convp3d
        #endif

        self.norm = InstanceNorm3d(outc)
    #end

    def conv2d(self, x):
        return self.conv1(x)
    #end

    def conv3d(self, x):
        return self.conv1(x)
    #end

    def convp3d(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    #end

    def forward(self, x):
        x = F.relu(x)
        x = self.conv(x)
        x = self.norm(x)
        return x
    #end
#end

class UpsampleBlock(nn.Module):
    def __init__(self, inc, outc):
        super(UpsampleBlock, self).__init__()

        self.up = Upsample(
            scale_factor=2,
            mode="trilinear"
        )
        self.conv = Conv3d(
            inc,
            outc,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.norm = InstanceNorm3d(outc)
    #end

    def forward(self, x):
        x = self.up(x)
        x = F.relu(x)
        x = self.conv(x)
        x = self.norm(x)
        return x
    #end
#end

class DownsampleBlock(nn.Module):
    def __init__(self, inc, outc):
        super(DownsampleBlock, self).__init__()

        self.conv = Conv3d(
            inc,
            outc,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self.norm = InstanceNorm3d(outc)
    #end

    def forward(self, x):
        x = F.relu(x)
        x = self.conv(x)
        x = self.norm(x)
        return x
    #end
#end

class Identity(nn.Module):
    def __init__(self, inc, outc, mode="3d"):
        super(Identity, self).__init__()

        self.prep = ConvBlock(inc, outc, kernel_size=1)
        self.conv = ConvBlock(outc, outc, kernel_size=3, mode=mode)
        self.post = ConvBlock(outc, outc, kernel_size=1)
    #end

    def forward(self, x):
        x = self.prep(x)
        x = self.conv(x)
        x = self.post(x)
        return x
    #end
#end

class IdentityDownsample(nn.Module):
    def __init__(self, inc, outc, mode="3d"):
        super(IdentityDownsample, self).__init__()

        self.prep = DownsampleBlock(inc, outc)
        self.conv = ConvBlock(outc, outc, kernel_size=3, mode=mode)
        self.post = ConvBlock(outc, outc, kernel_size=1)
    #end

    def forward(self, x):
        x = self.prep(x)
        x = self.conv(x)
        x = self.post(x)
        return x
    #end
#end

class Merge(nn.Module):
    def __init__(self, xc, prevc, outc, mode="3d"):
        super(Merge, self).__init__()

        self.prep0 = ConvBlock(xc,    outc, kernel_size=1)
        self.prep1 = ConvBlock(prevc, outc, kernel_size=1)

        self.conv0 = ConvBlock(outc, outc, kernel_size=3, mode=mode)
        self.conv1 = ConvBlock(outc, outc, kernel_size=3, mode=mode)

        self.post = ConvBlock(outc, outc, kernel_size=1)
    #end

    def forward(self, x, prev):
        x0 = self.prep0(x)
        x1 = self.prep1(prev)

        s = self.conv0(x0) + self.conv1(x1)

        s = self.post(s)
        return s
    #end
#end

class MergeUpsample(nn.Module):
    def __init__(self, xc, prevc, outc, mode="3d"):
        super(MergeUpsample, self).__init__()

        self.prep0 = UpsampleBlock(xc, outc)
        self.prep1 = ConvBlock(prevc, outc, kernel_size=1)

        self.conv0 = ConvBlock(outc, outc, kernel_size=3, mode=mode)
        self.conv1 = ConvBlock(outc, outc, kernel_size=3, mode=mode)

        self.post = ConvBlock(outc, outc, kernel_size=1)
    #end

    def forward(self, x, prev):
        x0 = self.prep0(x)
        x1 = self.prep1(prev)

        s = self.conv0(x0) + self.conv1(x1)

        s = self.post(s)
        return s
    #end
#end

class FirstStem(nn.Module):
    def __init__(self, inc, outc):
        super(FirstStem, self).__init__()

        filters = outc // 2

        self.conv1 = Conv3d(
            inc,
            filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.norm1 = InstanceNorm3d(filters)

        self.conv2 = Conv3d(
            filters,
            outc,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self.norm2 = InstanceNorm3d(outc)
    #end

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        return x
    #end
#end

class FinalStem(nn.Module):
    def __init__(self, inc, outc):
        super(FinalStem, self).__init__()

        filters = outc * 2

        self.conv1 = Conv3d(
            inc,
            filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.norm1 = InstanceNorm3d(filters)

        self.upsample = Upsample(
            scale_factor=2,
            mode="trilinear"
        )

        self.conv2 = Conv3d(
            filters,
            outc,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.norm2 = InstanceNorm3d(outc)
    #end

    def forward(self, x, skip):
        x = torch.cat((x, skip), dim=1)

        x = F.relu(x)
        x = self.conv1(x)
        x = self.norm1(x)

        x = self.upsample(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)

        return x
    #end
#end

class C2FNAS(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        init_filters=32,
        final_activation="softmax",
        apply_acti=False,
    ):
        super(C2FNAS, self).__init__()

        self.in_channels      = in_channels
        self.num_classes      = num_classes
        self.init_filters     = init_filters
        self.final_activation = final_activation
        self.apply_acti      = apply_acti

        if final_activation == "sigmoid" and num_classes != 1:
            raise ValueError("Output classes must be 1 when using sigmoid")
        #endif

        filters = init_filters

        self.first_stem = FirstStem(in_channels, filters * 2)

        self.encode_layers = nn.ModuleList([
            Identity(filters * 2, filters * 2, mode="2d"),
            Merge(filters * 2, filters * 2, filters * 2, mode="p3d"),
            IdentityDownsample(filters * 2, filters * 4, mode="3d"),
            Identity(filters * 4, filters * 4,  mode="3d"),
            IdentityDownsample(filters * 4, filters * 8,  mode="3d"),
            IdentityDownsample(filters * 8, filters * 16,  mode="3d")
        ])

        self.decode_layers = nn.ModuleList([
             MergeUpsample(filters * 16, filters * 8, filters * 8, mode="2d"),
             Identity(filters * 8, filters * 8, mode="3d"),
             MergeUpsample(filters * 8, filters * 4, filters * 4, mode="p3d"),
             MergeUpsample(filters * 4, filters * 2, filters * 2, mode="3d"),
             Identity(filters * 2, filters * 2, mode="2d"),
             Merge(filters * 2, filters * 2, filters * 2, mode="3d")
        ])

        # concatenate x and skip
        self.final_stem = FinalStem(filters * 4, filters)

        if final_activation.lower() == "sigmoid":
            self.final_conv = Conv3d(
                filters,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            self.output = Sigmoid()
        else:
            self.final_conv = Conv3d(
                filters,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            self.output = Softmax(dim=1)
        #endif
    #end

    def encoder(self, x):
        x = self.first_stem(x)

        skips = []

        stem = x
        x = self.encode_layers[0](x)
        x = self.encode_layers[1](x, stem)
        skips.append(x)

        x = self.encode_layers[2](x)
        x = self.encode_layers[3](x)
        skips.append(x)

        x = self.encode_layers[4](x)
        skips.append(x)

        x = self.encode_layers[5](x)

        return x, skips
    #end

    def decoder(self, x, skips):
        x = self.decode_layers[0](x, skips[2])

        x = self.decode_layers[1](x)
        x = self.decode_layers[2](x, skips[1])

        x = self.decode_layers[3](x, skips[0])

        prev = x
        x = self.decode_layers[4](x)
        s = self.decode_layers[5](x, prev)

        prev = x
        x = s
        x = self.final_stem(x, prev)

        return x
    #end

    def model(self, inp):
        x = inp['image'] if isinstance(inp, dict) else inp

        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        x = self.final_conv(x)
        if self.apply_acti:
            x = self.output(x)

        if isinstance(inp, dict):
            inp['prediction'] = x
            return inp
        else:
            return x
    #end

    def forward(self, x):
        return self.model(x)
    #end
#end
