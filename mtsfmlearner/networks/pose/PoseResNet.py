# Copyright NavInfo Europe 2023. Adapted from Packnet-sfm Toyota Research Institute. 

import torch
import torch.nn as nn

from mtsfmlearner.networks.layers.resnet.resnet_encoder import ResnetEncoder
from mtsfmlearner.networks.layers.resnet.pose_decoder import PoseDecoder

########################################################################################################################

class PoseResNet(nn.Module):
    """
    Pose network based on the ResNet architecture.

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
    def __init__(self, version=None, learn_intrinsics=False, **kwargs):
        super().__init__()
        assert version is not None, "PoseResNet needs a version"
        self.learn_intrinsics = learn_intrinsics
        num_layers = int(version[:-2])       # First two characters are the number of layers
        pretrained = version[-2:] == 'pt'    # If the last characters are "pt", use ImageNet pretraining
        assert num_layers in [18, 34, 50, 101], 'ResNet version {} not available'.format(num_layers)
        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained, num_input_images=2)
        self.decoder = PoseDecoder(self.encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2,
                                   learn_intrinsics=learn_intrinsics)

    def forward(self, target_image, ref_imgs):
        """
        Runs the network and returns predicted poses
        (1 for each reference image).
        """
        outputs_pose = []
        for i, ref_img in enumerate(ref_imgs):
            inputs = torch.cat([target_image, ref_img], 1)
            if self.learn_intrinsics:
                axisangle, translation, focal_length, offset = self.decoder([self.encoder(inputs)])
            else:
                axisangle, translation = self.decoder([self.encoder(inputs)])
            outputs_pose.append(torch.cat([translation[:, 0], axisangle[:, 0]], 2))
        pose = torch.cat(outputs_pose, 1)
        if self.learn_intrinsics:
            return pose, focal_length, offset
        else:
            return pose

########################################################################################################################

