# Copyright © NavInfo Europe 2023.

import torch.nn as nn
from functools import partial

from mtsfmlearner.networks.layers.resnet.depth_decoder import DepthDecoder
from mtsfmlearner.networks.layers.resnet.layers import disp_to_depth

from mtsfmlearner.networks.layers.transformers.transformer_encoder import TransformerEncoder
from mtsfmlearner.networks.layers.transformers.depth_decoder_dpt import DepthDecoderDpt
########################################################################################################################

class DepthMTSfMLearner(nn.Module):
    """
    Inverse depth network based on the MTSfMLearner architecture.

    Parameters
    ----------
    version : str
        an optional ImageNet pretrained flag added by the "pt" suffix
    kwargs : dict
        Extra parameters
    """
    def __init__(self, version=None, image_shape=(192, 640), num_scales=4,
                 transformer_model="deit-base", transformer_features=96, mim_strategy='', attack=False, **kwargs):
        super().__init__()
        assert version is not None, "DepthMTSfMLearner needs a version"

        self.attack = attack
        pt = version
        pretrained = pt == 'pt'
        self.encoder = TransformerEncoder(transformer_model=transformer_model,
                                          pretrained=pretrained,
                                          img_size=image_shape,
                                          mim_strategy=mim_strategy
                                          )

        self.decoder = DepthDecoderDpt(
            num_ch_enc=self.encoder.num_ch_enc,
            scales=range(num_scales),
            features=transformer_features
        )

        self.scale_inv_depth = partial(disp_to_depth, min_depth=0.1, max_depth=100.0)

    def forward(self, x, mask=None):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        x = self.encoder(x, mask)
        x = self.decoder(x)
        disps = [x[('disp', i)] for i in range(4)]

        if self.training or self.attack:
            return [self.scale_inv_depth(d)[0] for d in disps]
        else:
            return self.scale_inv_depth(disps[0])[0]

########################################################################################################################
