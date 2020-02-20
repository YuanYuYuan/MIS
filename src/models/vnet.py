import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBlock, AttentionBlock
from .parts import Normalization, CONV_LAYER, CONV_TRANSPOSE_LAYER


class InputConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        n_dim,
        norm,
    ):
        super().__init__()
        if isinstance(kernel_size, tuple):
            padding = tuple(k // 2 for k in kernel_size)
        else:
            padding = kernel_size // 2
        self.out_channels = out_channels
        self.conv = CONV_LAYER[n_dim](
            in_channels,
            out_channels,
            kernel_size,
            padding=padding
        )
        self.normalization = Normalization(n_dim, norm_type=norm)(out_channels)
        self.repeat_shape = (1, self.out_channels) + (1,) * n_dim

    def forward(self, x):
        output = self.normalization(self.conv(x))
        output += x.repeat(self.repeat_shape)
        output = F.relu(output)
        return output


class OutputConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        n_dim,
    ):

        super().__init__()
        self.conv = CONV_LAYER[n_dim](
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        return x


class VNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        base_channels=16,
        kernel_size=5,
        depth=3,
        n_labels=4,
        n_dim=2,
        in_shape=None,
        out_shape=None,
        norm='BatchNorm',
        attention=False,
        block_type=1,
    ):

        if isinstance(kernel_size, int):
            assert kernel_size % 2 == 1
        else:
            if isinstance(kernel_size, list):
                kernel_size = tuple(kernel_size)
            for k in kernel_size:
                assert k % 2 == 1
        super().__init__()

        self.input = InputConv(
            in_channels,
            base_channels,
            kernel_size,
            n_dim,
            norm
        )
        self.output = OutputConv(base_channels, n_labels, n_dim)
        self.n_dim = n_dim
        self.attention = attention

        # sanity check
        out_shape = tuple(out_shape) if out_shape else None
        in_shape = tuple(in_shape) if in_shape else None

        # output cropping
        if (out_shape is not None) and (out_shape != in_shape):
            crop_shape = out_shape
            left_crop = tuple(c//2 for c in crop_shape)
            right_crop = tuple(c - c//2 for c in crop_shape)
            center = tuple(s//2 for s in in_shape)
            self.crop_idx = (slice(None), slice(None),)  # [batch, channel]
            self.crop_idx += tuple(
                slice(c - l, c + r) for (c, l, r) in
                zip(center, left_crop, right_crop)
            )
        else:
            self.crop_idx = None

        # downward
        down_layers = []
        down_convs = []
        for d in range(depth):
            in_ch = base_channels * (2 ** d)
            out_ch = in_ch * 2

            # downsample
            down_convs.append(CONV_LAYER[n_dim](
                in_ch,
                out_ch,
                kernel_size=2,
                stride=2
            ))

            # encode
            down_layers.append(ConvBlock(
                out_ch,
                kernel_size,
                n_dim=n_dim,
                norm=norm,
                block_type=block_type,
            ))

        # upward
        up_layers = []
        up_convs = []
        for d in reversed(range(depth)):
            out_ch = base_channels * (2 ** d)
            in_ch = out_ch * 2

            # upsample
            up_convs.append(CONV_TRANSPOSE_LAYER[n_dim](
                in_ch,
                out_ch,
                kernel_size=2,
                stride=2
            ))

            # decode
            up_layers.append(ConvBlock(
                out_ch,
                kernel_size,
                n_dim=n_dim,
                norm=norm,
                block_type=block_type,
            ))

        # attention gates
        if attention:
            attention_gates = []
            for d in reversed(range(depth)):
                out_ch = base_channels * (2 ** d)
                in_ch = out_ch * 2
                attention_gates.append(AttentionBlock(
                    n_channels=out_ch,
                    n_dim=n_dim,
                    norm=norm
                ))
            self.attention_gates = nn.ModuleList(attention_gates)

        # wrap with module list
        self.down_layers = nn.ModuleList(down_layers)
        self.down_convs = nn.ModuleList(down_convs)
        self.up_layers = nn.ModuleList(up_layers)
        self.up_convs = nn.ModuleList(up_convs)

    def forward(self, x):
        x = self.input(x)

        # downward
        shortcuts = []
        for dc, dl in zip(self.down_convs, self.down_layers):
            shortcuts.append(x)
            x = dc(x)
            x = dl(x)

        # upward
        shortcuts.reverse()
        for idx in range(len(shortcuts)):
            uc = self.up_convs[idx]
            ul = self.up_layers[idx]
            sc = shortcuts[idx]

            if self.attention:
                sc = self.attention_gates[idx](sc, x)

            x = uc(x)
            if x.shape != sc.shape:
                padding = tuple()
                for i in range(self.n_dim):
                    padding += (0, sc.shape[-i-1]-x.shape[-i-1])
                x = F.pad(x, padding)
            x += sc
            x = ul(x)

        x = self.output(x)

        # output cropping
        if self.crop_idx is not None:
            return x[self.crop_idx]
        else:
            return x
