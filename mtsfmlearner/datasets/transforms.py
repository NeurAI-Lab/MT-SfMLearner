#  Copyright © NavInfo Europe 2023.
# Adapted from https://github.com/TRI-ML/packnet-sfm.

from functools import partial
from mtsfmlearner.datasets.augmentations import resize_image, resize_sample, \
    duplicate_sample, colorjitter_sample, to_tensor_sample
from mtsfmlearner.datasets import masked_image_modeling
from torch import tensor
########################################################################################################################

def train_transforms(sample, image_shape, jittering,
                     mim_strategy=None, mim_mask_size=None, mim_mask_ratio=None, mim_common=''):
    """
    Training data augmentation transformations

    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape
    jittering : tuple (brightness, contrast, saturation, hue)
        Color jittering parameters

    Returns
    -------
    sample : dict
        Augmented sample
    """
    if len(image_shape) > 0:
        sample = resize_sample(sample, image_shape)
    sample = duplicate_sample(sample)
    if len(jittering) > 0:
        sample = colorjitter_sample(sample, jittering)
    sample = to_tensor_sample(sample)

    if mim_strategy is None:
        mim_strategy = ['', '']
    if mim_mask_size is None:
        mim_mask_size = ['', '']
    if mim_mask_ratio is None:
        mim_mask_ratio = ['', '']

    masks = []

    mim_strategy_depth, mim_strategy_pose = mim_strategy
    mim_mask_size_depth, mim_mask_size_pose = mim_mask_size
    mim_mask_ratio_depth, mim_mask_ratio_pose = mim_mask_ratio
    # Mask for rgb
    if mim_strategy_depth != '':
        mim = masked_image_modeling.MIMTransform(img_size=image_shape,
                                                 mask_patch_size=mim_mask_size_depth,
                                                 mask_ratio=mim_mask_ratio_depth,
                                                 mask_strategy=mim_strategy_depth)
        mask = mim.mask_generator()
    else:
        mask = tensor([])
    masks.append(mask)
    if mim_common == 'all':
        masks = masks * (len(sample['rgb_context']) + 1)
        sample['mask'] = masks
        return sample

    # Mask for rgb contexts
    for i in range(len(sample['rgb_context'])):
        if mim_strategy_pose != '':
            mim = masked_image_modeling.MIMTransform(img_size=image_shape,
                                                     mask_patch_size=mim_mask_size_pose,
                                                     mask_ratio=mim_mask_ratio_pose,
                                                     mask_strategy=mim_strategy_pose)
            mask = mim.mask_generator()
        else:
            mask = tensor([])
        masks.append(mask)
    if mim_common == 'pose':
        masks[1:] = [masks[1]] * (len(sample['rgb_context']))
        sample['mask'] = masks
        return sample

    sample['mask'] = masks
    return sample

def validation_transforms(sample, image_shape):
    """
    Validation data augmentation transformations

    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape

    Returns
    -------
    sample : dict
        Augmented sample
    """
    if len(image_shape) > 0:
        sample['rgb'] = resize_image(sample['rgb'], image_shape)
        try:
            sample['rgb_context'] = [resize_image(img, image_shape) for img in sample['rgb_context']]
        except:
            pass
    sample = to_tensor_sample(sample)
    return sample

def test_transforms(sample, image_shape):
    """
    Test data augmentation transformations

    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape

    Returns
    -------
    sample : dict
        Augmented sample
    """
    if len(image_shape) > 0:
        sample['rgb'] = resize_image(sample['rgb'], image_shape)
        try:
            sample['rgb_context'] = [resize_image(img, image_shape) for img in sample['rgb_context']]
        except:
            pass
    sample = to_tensor_sample(sample)
    return sample

def get_transforms(mode, image_shape, jittering, mim_strategy='', **kwargs):
    """
    Get data augmentation transformations for each split

    Parameters
    ----------
    mode : str {'train', 'validation', 'test'}
        Mode from which we want the data augmentation transformations
    image_shape : tuple (height, width)
        Image dimension to reshape
    jittering : tuple (brightness, contrast, saturation, hue)
        Color jittering parameters

    Returns
    -------
        XXX_transform: Partial function
            Data augmentation transformation for that mode
    """
    if mode == 'train':
        return partial(train_transforms,
                       image_shape=image_shape,
                       jittering=jittering,
                       mim_strategy=mim_strategy,
                       **kwargs)
    elif mode == 'validation':
        return partial(validation_transforms,
                       image_shape=image_shape)
    elif mode == 'test':
        return partial(test_transforms,
                       image_shape=image_shape)
    else:
        raise ValueError('Unknown mode {}'.format(mode))

########################################################################################################################

