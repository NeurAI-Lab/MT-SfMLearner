# Copyright © NavInfo Europe 2023.
# Adapted from https://github.com/TRI-ML/packnet-sfm.

import random
import torch.nn as nn
from mtsfmlearner.utils.image import flip_model, interpolate_scales
from mtsfmlearner.geometry.pose import Pose
from mtsfmlearner.geometry.camera_utils import construct_K_from_predictions
from mtsfmlearner.utils.misc import make_list


class SfmModel(nn.Module):
    """
    Model class encapsulating a pose and depth networks.

    Parameters
    ----------
    depth_net : nn.Module
        Depth network to be used
    pose_net : nn.Module
        Pose network to be used
    rotation_mode : str
        Rotation mode for the pose network
    flip_lr_prob : float
        Probability of flipping when using the depth network
    upsample_depth_maps : bool
        True if depth map scales are upsampled to highest resolution
    kwargs : dict
        Extra parameters
    """
    def __init__(self, depth_net=None, pose_net=None,
                 rotation_mode='euler', flip_lr_prob=0.0,
                 upsample_depth_maps=False, learn_intrinsics=False, **kwargs):
        super().__init__()
        self.depth_net = depth_net
        self.pose_net = pose_net
        self.rotation_mode = rotation_mode
        self.flip_lr_prob = flip_lr_prob
        self.upsample_depth_maps = upsample_depth_maps
        self._logs = {}
        self._losses = {}
        self.learn_intrinsics = learn_intrinsics

        self._network_requirements = {
                'depth_net': True,  # Depth network required
                'pose_net': True,   # Pose network required
            }
        self._train_requirements = {
                'gt_depth': False,  # No ground-truth depth required
                'gt_pose': False,   # No ground-truth pose required
            }

    @property
    def logs(self):
        """Return logs."""
        return self._logs

    @property
    def losses(self):
        """Return metrics."""
        return self._losses

    def add_loss(self, key, val):
        """Add a new loss to the dictionary and detaches it."""
        self._losses[key] = val.detach()

    @property
    def network_requirements(self):
        """
        Networks required to run the model

        Returns
        -------
        requirements : dict
            depth_net : bool
                Whether a depth network is required by the model
            pose_net : bool
                Whether a depth network is required by the model
        """
        return self._network_requirements

    @property
    def train_requirements(self):
        """
        Information required by the model at training stage

        Returns
        -------
        requirements : dict
            gt_depth : bool
                Whether ground truth depth is required by the model at training time
            gt_pose : bool
                Whether ground truth pose is required by the model at training time
        """
        return self._train_requirements

    def add_depth_net(self, depth_net):
        """Add a depth network to the model"""
        self.depth_net = depth_net

    def add_pose_net(self, pose_net):
        """Add a pose network to the model"""
        self.pose_net = pose_net

    def compute_inv_depths(self, image, mask=None):
        """Computes inverse depth maps from single images"""
        # Randomly flip and estimate inverse depth maps
        flip_lr = random.random() < self.flip_lr_prob if self.training else False
        inv_depths = make_list(flip_model(self.depth_net, image, mask, flip_lr))
        # If upsampling depth maps
        if self.upsample_depth_maps:
            inv_depths = interpolate_scales(
                inv_depths, mode='nearest', align_corners=None)
        # Return inverse depth maps
        return inv_depths

    def compute_poses(self, image, contexts, masks=None):
        """Compute poses from image and a sequence of context images"""
        if masks is not None and all([m is not None for m in masks]):
            if self.learn_intrinsics:
                pose_vec, focal_length, offsets = self.pose_net(image, contexts, masks)
            else:
                pose_vec = self.pose_net(image, contexts, masks)
        else:
            if self.learn_intrinsics:
                pose_vec, focal_length, offsets = self.pose_net(image, contexts)
            else:
                pose_vec = self.pose_net(image, contexts)

        pose = [Pose.from_vec(pose_vec[:, i], self.rotation_mode)
                for i in range(pose_vec.shape[1])]
        if self.learn_intrinsics:
            return pose, focal_length, offsets
        else:
            return pose

    def forward(self, batch, return_logs=False):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        return_logs : bool
            True if logs are stored

        Returns
        -------
        output : dict
            Dictionary containing predicted inverse depth maps and poses
        """
        masks = {}
        if 'mask' in batch:
            for i, mask in enumerate(batch['mask']):
                if mask.nelement() > 0:
                    mask = mask.unsqueeze(1)
                else:
                    mask = None
                batch['mask'][i] = mask
            masks['rgb'] = batch['mask'][0]
            masks['rgb_context'] = batch['mask'][1:]
        else:
            masks['rgb'] = None
            masks['rgb_context'] = None



        # Generate inverse depth predictions
        inv_depths = self.compute_inv_depths(batch['rgb'], masks['rgb'])
        # Generate pose predictions if available
        pose = None
        K = batch['intrinsics']
        if 'rgb_context' in batch and self.pose_net is not None:
            if self.learn_intrinsics:
                pose, focal_lengths, offsets = self.compute_poses(batch['rgb'],
                                                                  batch['rgb_context'],
                                                                  masks['rgb_context'])

                K = construct_K_from_predictions(focal_lengths=focal_lengths,
                                                 offsets=offsets,
                                                 batch_size=len(batch['intrinsics']),
                                                 img_shape=tuple(batch['rgb'].shape[-2:]))
            else:
                pose = self.compute_poses(batch['rgb'],
                                          batch['rgb_context'],
                                          masks['rgb_context'])
        # Return output dictionary
        return {
            'inv_depths': inv_depths,
            'poses': pose,
            'intrinsics': K
        }
