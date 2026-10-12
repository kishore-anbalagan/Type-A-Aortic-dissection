import torch
import torch.nn as nn


class VNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder1 = self.block(1, 16)
        self.encoder2 = self.block(16, 32)
        self.encoder3 = self.block(32, 64)
        self.encoder4 = self.block(64, 128)

        self.pool = nn.MaxPool3d(2)

        self.bottleneck = self.block(128, 256)

        self.up4 = nn.ConvTranspose3d(
            256, 128, kernel_size=2, stride=2
        )

        self.decoder4 = self.block(256, 128)

        self.up3 = nn.ConvTranspose3d(
            128, 64, kernel_size=2, stride=2
        )

        self.decoder3 = self.block(128, 64)

        self.up2 = nn.ConvTranspose3d(
            64, 32, kernel_size=2, stride=2
        )

        self.decoder2 = self.block(64, 32)

        self.up1 = nn.ConvTranspose3d(
            32, 16, kernel_size=2, stride=2
        )

        self.decoder1 = self.block(32, 16)

        self.output = nn.Conv3d(
            16, 1, kernel_size=1
        )

    def block(self, in_channels, out_channels):

        return nn.Sequential(

            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        e1 = self.encoder1(x)

        e2 = self.encoder2(
            self.pool(e1)
        )

        e3 = self.encoder3(
            self.pool(e2)
        )

        e4 = self.encoder4(
            self.pool(e3)
        )

        b = self.bottleneck(
            self.pool(e4)
        )

        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.decoder4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.decoder3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.decoder2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.decoder1(d1)

        return self.output(d1)