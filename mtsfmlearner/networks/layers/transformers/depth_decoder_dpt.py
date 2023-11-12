# Copyright © NavInfo Europe 2023.

import torch.nn as nn
from .transformer_utils import _make_scratch, Interpolate, FeatureFusionBlock_custom
from mtsfmlearner.networks.layers.resnet.layers import Conv3x3


class DepthDecoderDpt(nn.Module):
    def __init__(self,
                 num_ch_enc=(96, 192, 384, 768),
                 scales=range(4),
                 features=256,
                 use_bn=True,
                 num_output_channels=1):

        super(DepthDecoderDpt, self).__init__()

        assert features in [64, 80, 96, 128, 256], 'Please choose transformer_features from [64, 80, 96, 128, 256]'

        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.num_output_channels = num_output_channels

        self.scratch = _make_scratch(
            self.num_ch_enc,
            out_shape=features,
            groups=1,
            expand=False
        )

        refinenets = []
        for i in range(4):
            refinenets.append(
                FeatureFusionBlock_custom(
                    features,
                    nn.ReLU(False),
                    deconv=False,
                    bn=use_bn,
                    expand=False,
                    align_corners=True
                )
            )
        self.refinenets = nn.Sequential(*refinenets)

        self.heads = nn.ModuleDict()
        for scale in scales:
            # Set head_convs = 2; heads = same
            self.heads[str(scale)] = nn.Sequential(
                nn.Conv2d(
                    features,
                    32,
                    kernel_size=3, stride=1, padding=1
                ),
                nn.ReLU(True),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(32, self.num_output_channels, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )

    def forward(self, input_features):
        self.outputs = {}

        for scale in self.scales[-1::-1]:
            layer_rn = self.scratch.layers_rn[scale](input_features[scale])
            if scale == 3:
                path = self.refinenets[scale](layer_rn)
            else:
                path = self.refinenets[scale](path, layer_rn)
            self.outputs[("disp", scale)] = self.heads[str(scale)](path)

        return self.outputs