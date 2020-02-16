import torch.nn as nn
import torch.nn.functional as F
from .parts import Normalization, CONV_LAYER, CONV_TRANSPOSE_LAYER


class ConvBlock(nn.Module):

    def __init__(
        self,
        n_channels,
        kernel_size=5,
        n_dim=2,
        norm='IN',
        block_type=1,
    ):
        super().__init__()
        padding = kernel_size // 2
        if block_type == 1:
            self.block = nn.Sequential(
                CONV_LAYER[n_dim](
                    n_channels,
                    n_channels,
                    kernel_size,
                    padding=padding
                ),
                Normalization(n_dim, norm_type=norm)(n_channels),
                nn.ReLU(),
                CONV_LAYER[n_dim](
                    n_channels,
                    n_channels,
                    kernel_size,
                    padding=padding
                ),
                Normalization(n_dim, norm_type=norm)(n_channels),
            )
        elif block_type == 2:
            self.block = nn.Sequential(
                Normalization(n_dim, norm_type=norm)(n_channels),
                nn.ReLU(),
                CONV_LAYER[n_dim](
                    n_channels,
                    n_channels,
                    kernel_size,
                    padding=padding
                ),
                Normalization(n_dim, norm_type=norm)(n_channels),
                nn.ReLU(),
                CONV_LAYER[n_dim](
                    n_channels,
                    n_channels,
                    kernel_size,
                    padding=padding
                ),
            )
        else:
            raise ValueError

    def forward(self, x):
        output = self.block(x)
        output += x
        output = F.relu(output)
        return output


class AttentionBlock(nn.Module):

    def __init__(
        self,
        n_channels,
        n_dim=3,
        norm='BN'
    ):
        super().__init__()
        self.n_dim = n_dim

        '''
        On upper featrue maps
            [C, W, H, D] -> [C, W, H, D]
        '''
        self.in1_conv = nn.Sequential(
            CONV_LAYER[n_dim](
                n_channels,
                n_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            Normalization(n_dim, norm_type=norm)(n_channels)
        )

        '''
        On gate signals from lower feature maps:
            [2*C, W, H, D] -> [C, W//2, H//2, D//2]
        '''
        self.in2_conv = nn.Sequential(
            CONV_TRANSPOSE_LAYER[n_dim](
                n_channels * 2,
                n_channels,
                kernel_size=2,
                stride=2
            ),
            Normalization(n_dim, norm_type=norm)(n_channels)
        )

        '''
        Attention gate
            [C, W, H, D] -> [1, W, H, D]
        '''
        self.attention = nn.Sequential(
            CONV_LAYER[n_dim](
                n_channels,
                1,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            Normalization(n_dim, norm_type=norm)(1),
            nn.Sigmoid()
        )

    def forward(self, in1, in2):
        '''
        in1: feature map of upper layer
        in2: feature map of lower layer
        The goal is to use the in2 as the attention gate and apply on the in1
        '''
        x1 = self.in1_conv(in1)
        x2 = self.in2_conv(in2)
        if x1.shape != x2.shape:
            padding = tuple()
            for i in range(self.n_dim):
                padding += (0, x1.shape[-i-1]-x2.shape[-i-1])
            x2 = F.pad(x2, padding)
        x = F.relu(x1+x2)
        x = self.attention(x)
        return in1 * x
