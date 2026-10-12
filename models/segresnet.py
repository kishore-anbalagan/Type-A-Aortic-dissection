import torch.nn as nn

from monai.networks.nets import SegResNet


class SegResNet3D(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = SegResNet(
            spatial_dims=3,
            init_filters=16,
            in_channels=1,
            out_channels=1,
            dropout_prob=0.0
        )

    def forward(self, x):

        return self.model(x)