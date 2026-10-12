from monai.networks.nets import UNet
import torch.nn as nn


class UNet3D(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

    def forward(self, x):
        return self.model(x)