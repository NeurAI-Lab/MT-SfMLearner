# Copyright © NavInfo Europe 2023.

import torch
import torch.nn as nn

from mtsfmlearner.networks.layers.transformers.transformer_encoder import TransformerEncoder
from mtsfmlearner.networks.layers.resnet.pose_decoder import PoseDecoder

########################################################################################################################

class PoseMTSfMLearner(nn.Module):
    """
    Pose network based on the MTSfMLearner architecture.

    Parameters
    ----------
    version : str
        an optional ImageNet pretrained flag added by the "pt" suffix
    kwargs : dict
        Extra parameters
    """
    def __init__(self, version=None,  image_shape=(192, 640), transformer_model="deit-base",
                 learn_intrinsics=False, mim_strategy='', cat_dim=1, **kwargs):
        super().__init__()
        assert version is not None, "PoseMTSfMLearner needs a version"
        self.learn_intrinsics = learn_intrinsics

        assert cat_dim in [1, 2, 3], "Can only concatenate along channels(1), height(2), or width(3)"

        pt = version
        pretrained = pt == 'pt'

        self.cat_dim = cat_dim

        self.encoder = TransformerEncoder(transformer_model=transformer_model,
                                          pretrained=pretrained,
                                          img_size=image_shape,
                                          num_input_images=2,
                                          mim_strategy=mim_strategy,
                                          cat_dim=cat_dim
                                          )

        self.decoder = PoseDecoder(self.encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2,
                                   learn_intrinsics=learn_intrinsics)

    def forward(self, target_image, ref_imgs, masks=None):
        """
        Runs the network and returns predicted poses
        (1 for each reference image).
        """
        outputs_pose = []
        if masks is None:
            masks = [None] * len(ref_imgs)

        for i, (ref_img, mask) in enumerate(zip(ref_imgs, masks)):
            inputs = torch.cat([target_image, ref_img], self.cat_dim)#1)
            if self.learn_intrinsics:
                axisangle, translation, focal_length, offset = self.decoder([self.encoder(inputs, mask)])
            else:
                axisangle, translation = self.decoder([self.encoder(inputs, mask)])
            outputs_pose.append(torch.cat([translation[:, 0], axisangle[:, 0]], 2))
        pose = torch.cat(outputs_pose, 1)
        if self.learn_intrinsics:
            return pose, focal_length, offset
        else:
            return pose

########################################################################################################################

