# Copyright © NavInfo Europe 2023.

from __future__ import absolute_import, division, print_function

import yaml
import numpy as np

import torch
import torch.nn as nn
from .transformers_deit import *
from .transformers_pvt import *
from .transformer_utils import trunc_normal_


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


def load_interpolated_encoder(checkpoint, model, num_input_images=1):
    #checkpoint_model = checkpoint['model']
    if num_input_images > 1:
        checkpoint['encoder.patch_embed.proj.weight'] = torch.cat(
            [checkpoint['encoder.patch_embed.proj.weight']] * num_input_images, 1
        ) / num_input_images
    state_dict = model.state_dict()
    for k in ['encoder.head.weight', 'encoder.head.bias', 'encoder.head_dist.weight', 'encoder.head_dist.bias']:
        if k in checkpoint and k in state_dict and checkpoint[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint[k]

    # interpolate position embedding
    pos_embed_checkpoint = checkpoint['encoder.pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.encoder.patch_embed.num_patches
    num_extra_tokens = model.encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    #orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    orig_size = (
            (pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (
                checkpoint["height"] // model.encoder.patch_embed.patch_size[0]),
            (pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (
                        checkpoint["width"] // model.encoder.patch_embed.patch_size[0])
    )
    # height (== width) for the new position embedding
    #new_size = int(num_patches ** 0.5)
    new_size = (model.encoder.patch_embed.img_size[0] // model.encoder.patch_embed.patch_size[0],
                model.encoder.patch_embed.img_size[1] // model.encoder.patch_embed.patch_size[1])
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size[0], orig_size[1], embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, #size=(new_size, new_size),
        size=new_size, mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint['encoder.pos_embed'] = new_pos_embed

    model.load_state_dict(checkpoint, strict=False)


def load_interpolated_pose_encoder(checkpoint, model, size, num_input_images=1):

    width, height = size

    state_dict = model.state_dict()

    for k in ['encoder.head.weight', 'encoder.head.bias', 'encoder.head_dist.weight', 'encoder.head_dist.bias']:
        if k in checkpoint and k in state_dict and checkpoint[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint[k]

    # interpolate position embedding
    pos_embed_checkpoint = checkpoint['encoder.pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.encoder.patch_embed.num_patches
    num_extra_tokens = model.encoder.pos_embed.shape[-2] - num_patches
    orig_size = (
            (pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (
                height // model.encoder.patch_embed.patch_size[0]),
            (pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (
                        width // model.encoder.patch_embed.patch_size[0])
    )
    new_size = (model.encoder.patch_embed.img_size[0] // model.encoder.patch_embed.patch_size[0],
                model.encoder.patch_embed.img_size[1] // model.encoder.patch_embed.patch_size[1])
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size[0], orig_size[1], embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, #size=(new_size, new_size),
        size=new_size, mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint['encoder.pos_embed'] = new_pos_embed

    model.load_state_dict(checkpoint, strict=False)


class TransformerEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, transformer_model, pretrained, img_size=(192, 640), num_input_images=1, mim_strategy='',
                 cat_dim=1):
        super(TransformerEncoder, self).__init__()

        # decoder based on num_input_images.
        # Inspired by DPT for depth and by MD2 for pose
        assert num_input_images in [1, 2], 'The num_input_images to transformers is either 1 (depth) or 2 (pose)'
        self.num_input_images = num_input_images

        if "pvt" not in transformer_model:
            if self.num_input_images == 2:
                self.num_ch_enc = np.array([64, 64, 128, 256, 512])
            else:
                self.num_ch_enc = np.array([96, 192, 384, 768])
            self.num_ch_enc[1:] *= 4
        else:
            self.num_ch_enc = np.array([64, 128, 320, 512])

        if num_input_images > 1 and cat_dim == 1:
            self.register_buffer('img_mean',
                                 torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406]).view(1, -1, 1, 1))
            self.register_buffer('img_std',
                                 torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225]).view(1, -1, 1, 1))
        else:
            self.register_buffer('img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
            self.register_buffer('img_std', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

        transformers = {
            "deit-base": deit_base_patch16_224,
            "pvt-b4": pvt_b4
        }

        # self.num_input_images == 1
        self.hooks = [2, 5, 8, 11]

        if transformer_model not in transformers:
            raise ValueError("{} is not a valid transformer model".format(transformer_model))

        self.encoder = transformers[transformer_model](img_size=img_size,
                                                       pretrained=pretrained,
                                                       num_input_images=num_input_images,
                                                       cat_dim=cat_dim)
        self.reassemble = nn.ModuleList()

        if not isinstance(self.encoder, PyramidVisionTransformerV2):
            if cat_dim == 1:
                og_img_size = torch.Size([img_size[0] // self.encoder.patch_embed.patch_size[0],
                                          img_size[1] // self.encoder.patch_embed.patch_size[0]])
            else:
                og_img_size = torch.Size([img_size[i] * 2 // self.encoder.patch_embed.patch_size[0] if i == cat_dim - 2
                                          else img_size[i] // self.encoder.patch_embed.patch_size[0]
                                          for i in range(len(img_size))])

        if not isinstance(self.encoder, PyramidVisionTransformerV2):
            for i, num_ch_enc in enumerate(self.num_ch_enc):
                in_channels = self.encoder.embed_dim

                modules = [
                    Transpose(1, 2),
                    nn.Unflatten(2,
                                 og_img_size,
                                 # torch.Size([img_size[0] // self.encoder.patch_embed.patch_size[0],
                                 #             img_size[1] // self.encoder.patch_embed.patch_size[0]])
                                 ),
                    nn.Conv2d(
                        in_channels=in_channels, #self.encoder.embed_dim,
                        out_channels=num_ch_enc,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                ]
                if self.num_input_images == 1:
                    if i < len(self.num_ch_enc) - 1:
                        if (len(self.num_ch_enc) - i - 2) != 0:
                            modules.append(
                                nn.ConvTranspose2d(
                                    in_channels=num_ch_enc,
                                    out_channels=num_ch_enc,
                                    kernel_size=2 ** (len(self.num_ch_enc) - i - 2),
                                    stride=2 ** (len(self.num_ch_enc) - i - 2),
                                    padding=0,
                                    bias=True,
                                    dilation=1,
                                    groups=1
                                )
                            )
                    else:
                        modules.append(
                            nn.Conv2d(
                                in_channels=num_ch_enc,
                                out_channels=num_ch_enc,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=True,
                                dilation=1,
                                groups=1
                            )
                        )
                else:
                    if i < len(self.num_ch_enc) - 1:
                        modules.append(
                            nn.ConvTranspose2d(
                                in_channels=num_ch_enc,
                                out_channels=num_ch_enc,
                                kernel_size=2 ** (len(self.num_ch_enc) - i - 2),
                                stride=2 ** (len(self.num_ch_enc) - i - 2),
                                padding=0,
                                bias=True,
                                dilation=1,
                                groups=1
                            )
                        )
                self.reassemble.append(nn.Sequential(*modules))

        if any([strategy != '' for strategy in mim_strategy]):
            self.encoder.mask_token = nn.Parameter(torch.zeros(1, 1, self.encoder.embed_dim))
            trunc_normal_(self.encoder.mask_token, std=.02)

    def forward(self, input_image, mask=None):
        self.features = []
        x = (input_image - self.img_mean) / self.img_std

        if isinstance(self.encoder, PyramidVisionTransformerV2):
            B = x.shape[0]

            for i in range(self.encoder.num_stages):
                patch_embed = getattr(self.encoder, f"patch_embed{i + 1}")
                block = getattr(self.encoder, f"block{i + 1}")
                norm = getattr(self.encoder, f"norm{i + 1}")
                x, H, W = patch_embed(x)
                for blk in block:
                    x = blk(x, H, W)
                x = norm(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                self.features.append(x)

        else:

            if self.num_input_images == 2:
                B = x.shape[0]
                x = self.encoder.patch_embed(x)

                if mask is not None:
                    B, L, _ = x.shape
                    mask_token = self.encoder.mask_token.expand(B, L, -1)
                    w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
                    x = x * (1 - w) + mask_token * w

                cls_tokens = self.encoder.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                x = torch.cat((cls_tokens, x), dim=1)
                x = x + self.encoder.pos_embed

                x = self.encoder.pos_drop(x)

                for blk in self.encoder.blocks:
                    x = blk(x)

                x = self.encoder.norm(x)
                # Set read to ignore
                feats = x[:, 1:]

                for i in range(len(self.num_ch_enc)):
                    if i < len(self.num_ch_enc) - 1:
                        self.features.append(
                            self.reassemble[i](feats)
                        )
                    else:
                        self.features.append(
                            nn.functional.interpolate(
                                self.reassemble[i](feats),
                                scale_factor=2 ** (len(self.num_ch_enc) - i - 2),
                                mode='bilinear',
                                align_corners=True
                            )
                        )
            else:
                B = x.shape[0]
                x = self.encoder.patch_embed(x)

                if mask is not None:
                    B, L, _ = x.shape
                    mask_token = self.encoder.mask_token.expand(B, L, -1)
                    w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
                    x = x * (1 - w) + mask_token * w

                cls_tokens = self.encoder.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                x = torch.cat((cls_tokens, x), dim=1)
                x = x + self.encoder.pos_embed

                x = self.encoder.pos_drop(x)

                for hook, blk in enumerate(self.encoder.blocks):
                    x = blk(x)
                    if hook in self.hooks:
                        ind = self.hooks.index(hook)
                        if self.num_input_images == 2:
                            if ind < len(self.num_ch_enc) - 1:
                                self.features.append(
                                    self.reassemble[ind](x[:, 1:])
                                )
                            else:
                                self.features.append(
                                    nn.functional.interpolate(
                                        self.reassemble[ind](x[:, 1:]),
                                        scale_factor=2 ** (len(self.num_ch_enc) - ind - 2),
                                        mode='bilinear',
                                        align_corners=True
                                    )
                                )
                        else:
                            if hook != self.hooks[-1]:
                                self.features.append(self.reassemble[ind](x[:, 1:]))

                x = self.encoder.norm(x)

                feats = x[:, 1:]

                self.features.append(self.reassemble[-1](feats))

        return self.features
