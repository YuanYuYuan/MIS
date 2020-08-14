import torch.nn.functional as F
import torch.nn as nn
import torch

# This code is adapted from https://github.com/milesial/Pytorch-UNet,
# and modified to satisfy the requirements for pytorch-curv.


class UNet2D(nn.Module):

    def __init__(self, n_channels=1, n_classes=4):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # XXX: for SSO
        self.num_classes = n_classes


        self.inc1 = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        self.inc_bn1 = nn.BatchNorm2d(64)
        self.inc2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.inc_bn2 = nn.BatchNorm2d(64)

        self.down11 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.down1_bn1 = nn.BatchNorm2d(128)
        self.down12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.down1_bn2 = nn.BatchNorm2d(128)

        self.down21 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.down2_bn1 = nn.BatchNorm2d(256)
        self.down22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.down2_bn2 = nn.BatchNorm2d(256)

        self.down31 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.down3_bn1 = nn.BatchNorm2d(512)
        self.down32 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.down3_bn2 = nn.BatchNorm2d(512)

        self.down41 = nn.Conv2d(512, 1024 // 2, kernel_size=3, padding=1)
        self.down4_bn1 = nn.BatchNorm2d(512)
        self.down42 = nn.Conv2d(1024 // 2, 1024 // 2, kernel_size=3, padding=1)
        self.down4_bn2 = nn.BatchNorm2d(512)

        self.up11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.up1_bn1 = nn.BatchNorm2d(512)
        self.up12 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.up1_bn2 = nn.BatchNorm2d(256)

        self.up21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.up2_bn1 = nn.BatchNorm2d(256)
        self.up22 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.up2_bn2 = nn.BatchNorm2d(128)

        self.up31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.up3_bn1 = nn.BatchNorm2d(128)
        self.up32 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.up3_bn2 = nn.BatchNorm2d(64)

        self.up41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.up4_bn1 = nn.BatchNorm2d(64)
        self.up42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.up4_bn2 = nn.BatchNorm2d(64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, inp):
        x = inp['image'] if isinstance(inp, dict) else inp
        x = self.inc1(x)
        x = self.inc_bn1(x)
        x = F.relu(x, inplace=True)
        x = self.inc2(x)
        x = self.inc_bn2(x)
        x1 = F.relu(x, inplace=True)

        x = F.max_pool2d(x1, 2)
        x = self.down11(x)
        x = self.down1_bn1(x)
        x = F.relu(x, inplace=True)
        x = self.down12(x)
        x = self.down1_bn2(x)
        x2 = F.relu(x, inplace=True)

        x = F.max_pool2d(x2, 2)
        x = self.down21(x)
        x = self.down2_bn1(x)
        x = F.relu(x, inplace=True)
        x = self.down22(x)
        x = self.down2_bn2(x)
        x3 = F.relu(x, inplace=True)

        x = F.max_pool2d(x3, 2)
        x = self.down31(x)
        x = self.down3_bn1(x)
        x = F.relu(x, inplace=True)
        x = self.down32(x)
        x = self.down3_bn2(x)
        x4 = F.relu(x, inplace=True)

        x = F.max_pool2d(x4, 2)
        x = self.down41(x)
        x = self.down4_bn1(x)
        x = F.relu(x, inplace=True)
        x = self.down42(x)
        x = self.down4_bn2(x)
        x5 = F.relu(x, inplace=True)

        x = F.upsample_bilinear(x5, scale_factor=2)
        diffY = x4.size()[2] - x.size()[2]
        diffX = x4.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x4, x], dim=1)
        x = self.up11(x)
        x = self.up1_bn1(x)
        x = F.relu(x, inplace=True)
        x = self.up12(x)
        x = self.up1_bn2(x)
        x = F.relu(x, inplace=True)

        x = F.upsample_bilinear(x, scale_factor=2)
        diffY = x3.size()[2] - x.size()[2]
        diffX = x3.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x3, x], dim=1)
        x = self.up21(x)
        x = self.up2_bn1(x)
        x = F.relu(x, inplace=True)
        x = self.up22(x)
        x = self.up2_bn2(x)
        x = F.relu(x, inplace=True)

        x = F.upsample_bilinear(x, scale_factor=2)
        diffY = x2.size()[2] - x.size()[2]
        diffX = x2.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x], dim=1)
        x = self.up31(x)
        x = self.up3_bn1(x)
        x = F.relu(x, inplace=True)
        x = self.up32(x)
        x = self.up3_bn2(x)
        x = F.relu(x, inplace=True)

        x = F.upsample_bilinear(x, scale_factor=2)
        diffY = x1.size()[2] - x.size()[2]
        diffX = x1.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x], dim=1)
        x = self.up41(x)
        x = self.up4_bn1(x)
        x = F.relu(x, inplace=True)
        x = self.up42(x)
        x = self.up4_bn2(x)
        x = F.relu(x, inplace=True)

        logits = self.outc(x)

        if isinstance(inp, dict):
            inp['prediction'] = logits
            return inp
        else:
            return logits
