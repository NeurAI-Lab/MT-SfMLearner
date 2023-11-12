# Copyright © NavInfo Europe 2023.
# Adapted from SimMIM and BEIT.

import numpy as np
import random
import math

class MaskGenerator:
    def __init__(self, img_shape=(192, 640), mask_patch_size=32, model_patch_size=4, mask_ratio=0.25,
                 mask_strategy='random'):
        self.img_shape = img_shape
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy

        assert self.img_shape[0] % self.mask_patch_size == 0
        assert self.img_shape[1] % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.token_shape = np.zeros(len(self.img_shape), dtype=int)
        self.token_shape[0] = self.img_shape[0] // self.mask_patch_size
        self.token_shape[1] = self.img_shape[1] // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.token_shape[0] * self.token_shape[1]
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def random_masking(self):
        mask = np.zeros(shape=self.token_count, dtype=int)
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask[mask_idx] = 1

        mask = mask.reshape((self.token_shape[0], self.token_shape[1]))
        return mask

    def blockwise_masking(self, min_num_mask_patches=16,
                          min_blockwise_aspect=0.3):
        mask = np.zeros(shape=self.token_count, dtype=int)
        mask = mask.reshape((self.token_shape[0], self.token_shape[1]))
        num_tokens_masked = 0
        NUM_TRIES = 10
        max_blockwise_aspect = 1 / min_blockwise_aspect
        log_aspect_ratio = (math.log(min_blockwise_aspect), math.log(max_blockwise_aspect))
        while num_tokens_masked < self.mask_count:
            max_mask_patches = self.mask_count - num_tokens_masked

            delta = 0
            for attempt in range(NUM_TRIES):
                target_area = random.uniform(min_num_mask_patches, max_mask_patches)
                aspect_ratio = math.exp(random.uniform(*log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if h < self.token_shape[0] and w < self.token_shape[1]:
                    top = random.randint(0, self.token_shape[0] - h)
                    left = random.randint(0, self.token_shape[1] - w)

                    num_masked = mask[top: top + h, left: left + w].sum()
                    # Overlap
                    if 0 < h * w - num_masked <= max_mask_patches:
                        for i in range(top, top + h):
                            for j in range(left, left + w):
                                if mask[i, j] == 0:
                                    mask[i, j] = 1
                                    delta += 1

                    if delta > 0:
                        break

            if delta == 0:
                break
            else:
                num_tokens_masked += delta

        return mask

    def __call__(self):
        if self.mask_strategy == 'random':
            mask = self.random_masking()
        elif self.mask_strategy == 'blockwise':
            mask = self.blockwise_masking()
        else:
            raise NotImplementedError
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask


class MIMTransform:
    def __init__(self, img_size, mask_patch_size, mask_ratio, mask_strategy='random'):

        model_patch_size = 16
        self.mask_generator = MaskGenerator(
            img_shape=img_size,
            mask_patch_size=mask_patch_size,
            model_patch_size=model_patch_size,
            mask_ratio=mask_ratio,
            mask_strategy=mask_strategy
        )

    def __call__(self):
        mask = self.mask_generator()
        return mask