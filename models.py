# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F


from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384', 'deit_small_hi_patch16_224'
]


class HiVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        #self.hi_token_1 = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        #self.hi_token_2 = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        #self.hi_token_3 = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        #self.hi_token_4 = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.prototype_0 = nn.Linear(self.embed_dim, 2 + 1, bias = False)
        self.prototype_1 = nn.Linear(self.embed_dim, 2 + 1, bias = False)
        self.prototype_2 = nn.Linear(self.embed_dim, 2 + 1, bias = False)
        self.prototype_3 = nn.Linear(self.embed_dim, 4 + 1, bias = False)
        self.prototype_4 = nn.Linear(self.embed_dim, 9 + 1, bias = False)
        self.prototype_5 = nn.Linear(self.embed_dim, 16 + 1, bias = False)
        self.prototype_6 = nn.Linear(self.embed_dim, 25 + 1, bias = False)
        self.prototype_7 = nn.Linear(self.embed_dim, 49 + 1, bias = False)
        self.prototype_8 = nn.Linear(self.embed_dim, 90 + 1, bias = False)
        self.prototype_9 = nn.Linear(self.embed_dim, 170 + 1, bias = False)
        self.prototype_10 = nn.Linear(self.embed_dim, 406 + 1, bias = False)
        
        self.matrix_0 = nn.Linear(self.embed_dim, 2 + 1, bias = False)
        self.matrix_1 = nn.Linear(self.embed_dim, 2 + 1, bias = False)
        self.matrix_2 = nn.Linear(self.embed_dim, 2 + 1, bias = False)
        self.matrix_3 = nn.Linear(self.embed_dim, 4 + 1, bias = False)
        self.matrix_4 = nn.Linear(self.embed_dim, 9 + 1, bias = False)
        self.matrix_5 = nn.Linear(self.embed_dim, 16 + 1, bias = False)
        self.matrix_6 = nn.Linear(self.embed_dim, 25 + 1, bias = False)
        self.matrix_7 = nn.Linear(self.embed_dim, 49 + 1, bias = False)
        self.matrix_8 = nn.Linear(self.embed_dim, 90 + 1, bias = False)
        self.matrix_9 = nn.Linear(self.embed_dim, 170 + 1, bias = False)
        self.matrix_10 = nn.Linear(self.embed_dim, 406 + 1, bias = False)
        
        #trunc_normal_(self.hi_token_1, std=.02)
        #trunc_normal_(self.hi_token_2, std=.02)
        #trunc_normal_(self.hi_token_3, std=.02)
        #trunc_normal_(self.hi_token_4, std=.02)
        
        trunc_normal_(self.pos_embed, std=.02)
        self.prototype_0.apply(self._init_weights)
        self.prototype_1.apply(self._init_weights)
        self.prototype_2.apply(self._init_weights)
        self.prototype_3.apply(self._init_weights)
        self.prototype_4.apply(self._init_weights)
        self.prototype_5.apply(self._init_weights)
        self.prototype_6.apply(self._init_weights)
        self.prototype_7.apply(self._init_weights)
        self.prototype_8.apply(self._init_weights)
        self.prototype_9.apply(self._init_weights)
        self.prototype_10.apply(self._init_weights)
        
        self.matrix_0.apply(self._init_weights)
        self.matrix_1.apply(self._init_weights)
        self.matrix_2.apply(self._init_weights)
        self.matrix_3.apply(self._init_weights)
        self.matrix_4.apply(self._init_weights)
        self.matrix_5.apply(self._init_weights)
        self.matrix_6.apply(self._init_weights)
        self.matrix_7.apply(self._init_weights)
        self.matrix_8.apply(self._init_weights)
        self.matrix_9.apply(self._init_weights)
        self.matrix_10.apply(self._init_weights)

    def forward_features(self, x, targets_list):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the hi_token
        B = x.shape[0]
        
        x = self.patch_embed(x)
        num_patch = x.shape[1]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        
        x = torch.cat((cls_tokens, x), dim=1)
        ori = x.shape[1]

        x = x + self.pos_embed
        x = self.pos_drop(x)

        prototypes = [self.prototype_0, self.prototype_1, self.prototype_2, self.prototype_3, \
                      self.prototype_4, self.prototype_5, self.prototype_6, self.prototype_7, \
                      self.prototype_8, self.prototype_9, self.prototype_10]
        
        matrices = [self.matrix_0, self.matrix_1, self.matrix_2, self.matrix_3, \
                    self.matrix_4, self.matrix_5, self.matrix_6, self.matrix_7, \
                    self.matrix_8, self.matrix_9, self.matrix_10]
        
        
        self.head = self.head

        emb_dim = x.shape[-1]
        proto_cls_probs = []
        for b in range(len(self.blocks)-1):
            if b >= 10: #change 2
                x_with_prototype = torch.cat((x, prototypes[b].weight.expand(B,-1,-1)), dim=1)
                x_with_prototype = self.blocks[b](x_with_prototype)
                x = x_with_prototype[:, :ori, :]
                #proto_cls_prob = torch.sigmoid(x_with_prototype[:, ori:, :] @ self.matrix.weight[b])
                proto_cls_prob = x_with_prototype[:, ori:, :] @ (matrices[b].weight.T)
                proto_cls_probs.append(torch.diagonal(proto_cls_prob, dim1=-2, dim2=-1))
            else:
                x = self.blocks[b](x)
                proto_cls_prob = (prototypes[b].weight.expand(B,-1,-1)) @ (matrices[b].weight.T)
                proto_cls_probs.append(torch.diagonal(proto_cls_prob, dim1=-2, dim2=-1))
        
        x = self.blocks[-1](x)
        x = self.norm(x)
        
        return proto_cls_probs, x[:, 0]

    def forward(self, x, targets_list):
        proto_cls_probs, x_final = self.forward_features(x, targets_list)
        
        x_final = self.head(x_final)
        if not self.training:
            # during inference, return the last classifier predictions
            return x_final
        
        if self.training:
            return proto_cls_probs, x_final

class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2
        
        
@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_small_hi_patch16_224(pretrained=False, **kwargs):
    model = HiVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        print("Wait for implementation!")
    return model

@register_model
def deit_base_hi_patch16_224(pretrained=False, **kwargs):
    model = HiVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        print("Wait for implementation!")
    return model

@register_model
def deit_base_hi_patch16_384(pretrained=False, **kwargs):
    model = HiVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        print("Wait for implementation!")
    return model