import torch.nn as nn


class Classifier(nn.Module):

    def __init__(self, out_channels=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(3, 2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(3, 2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(3, 2),

            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, out_channels),
        )

    def forward(self, x):
        # print(self.cnn(x).shape)
        return self.cnn(x)
