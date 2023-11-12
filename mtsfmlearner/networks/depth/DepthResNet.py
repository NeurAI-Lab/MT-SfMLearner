# Copyright NavInfo Europe 2023. Adapted from Packnet-sfm Toyota Research Institute. 

import torch.nn as nn
from functools import partial

from mtsfmlearner.networks.layers.resnet.resnet_encoder import ResnetEncoder
from mtsfmlearner.networks.layers.resnet.depth_decoder import DepthDecoder
from mtsfmlearner.networks.layers.resnet.layers import disp_to_depth

########################################################################################################################

class DepthResNet(nn.Module):
    """
    Inverse depth network based on the ResNet architecture.

    Parameters
    ----------
    version : str
        Has a XY format, where:
        X is the number of residual layers [18, 34, 50] and
        Y is an optional ImageNet pretrained flag added by the "pt" suffix
        Example: "18pt" initializes a pretrained ResNet18, and "34" initializes a ResNet34 from scratch
    kwargs : dict
        Extra parameters
    """
    def __init__(self, version=None, attack=False, **kwargs):
        super().__init__()
        assert version is not None, "DispResNet needs a version"

        self.attack = attack
        num_layers = int(version[:-2])       # First two characters are the number of layers
        pretrained = version[-2:] == 'pt'    # If the last characters are "pt", use ImageNet pretraining
        assert num_layers in [18, 34, 50, 101], 'ResNet version {} not available'.format(num_layers)

        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained)
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc)
        self.scale_inv_depth = partial(disp_to_depth, min_depth=0.1, max_depth=100.0)

    def forward(self, x):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        x = self.encoder(x)
        x = self.decoder(x)
        disps = [x[('disp', i)] for i in range(4)]

        if self.training or self.attack:
            return [self.scale_inv_depth(d)[0] for d in disps]
        else:
            return self.scale_inv_depth(disps[0])[0]

########################################################################################################################
