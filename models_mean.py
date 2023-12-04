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
        
        self.hi_token_1 = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.hi_token_2 = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.hi_token_3 = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        #self.hi_token_4 = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1 + 3, self.embed_dim))
        self.head_0 = nn.Linear(self.embed_dim, 2 + 1, bias = False)
        self.head_1 = nn.Linear(self.embed_dim, 2 + 1, bias = False)
        self.head_2 = nn.Linear(self.embed_dim, 2 + 1, bias = False)
        self.head_3 = nn.Linear(self.embed_dim, 4 + 1, bias = False)
        self.head_4 = nn.Linear(self.embed_dim, 9 + 1, bias = False)
        self.head_5 = nn.Linear(self.embed_dim, 16 + 1, bias = False)
        self.head_6 = nn.Linear(self.embed_dim, 25 + 1, bias = False)
        self.head_7 = nn.Linear(self.embed_dim, 49 + 1, bias = False)
        self.head_8 = nn.Linear(self.embed_dim, 90 + 1, bias = False)
        self.head_9 = nn.Linear(self.embed_dim, 170 + 1, bias = False)
        self.head_10 = nn.Linear(self.embed_dim, 406 + 1, bias = False)
        
        trunc_normal_(self.hi_token_1, std=.02)
        trunc_normal_(self.hi_token_2, std=.02)
        trunc_normal_(self.hi_token_3, std=.02)
        #trunc_normal_(self.hi_token_4, std=.02)
        
        trunc_normal_(self.pos_embed, std=.02)
        self.head_0.apply(self._init_weights)
        self.head_1.apply(self._init_weights)
        self.head_2.apply(self._init_weights)
        self.head_3.apply(self._init_weights)
        self.head_4.apply(self._init_weights)
        self.head_5.apply(self._init_weights)
        self.head_6.apply(self._init_weights)
        self.head_7.apply(self._init_weights)
        self.head_8.apply(self._init_weights)
        self.head_9.apply(self._init_weights)
        self.head_10.apply(self._init_weights)

    def forward_features(self, x, targets_list):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the hi_token
        B = x.shape[0]
        
        x = self.patch_embed(x)
        num_patch = x.shape[1]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        
        hi_token_1 = self.hi_token_1.expand(B, -1, -1)
        hi_token_2 = self.hi_token_2.expand(B, -1, -1)
        hi_token_3 = self.hi_token_3.expand(B, -1, -1)
        #hi_token_4 = self.hi_token_4.expand(B, -1, -1)
        
        
        x = torch.cat((cls_tokens, hi_token_1, hi_token_2, hi_token_3, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        heads = [self.head_0, self.head_1, self.head_2, self.head_3, self.head_4, self.head_5, \
                 self.head_6, self.head_7, self.head_8, self.head_9, self.head_10, self.head]

        xs = []
        emb_dim = x.shape[-1]
        for b in range(len(self.blocks)):
            #print(b)
            x = self.blocks[b](x)
            xs.append(self.norm(x))
            
            if (b > 7) and (b != 11):
                #Stay tuned, Not sure
                #targets_list[b]: shape B*(Nb_classes) (Include two images mixing-up)
                #heads[b].weight: shape Nb_classes*embed_dim
                #We use mask to filter out the missing labels
                #'''

                #print("We are on the %d layer!"%b)
                #print(targets_list[b])
                #print(masks_list[b])
                #'''
                if self.training:
                    select_feature_with_weight = targets_list[b] @ heads[b].weight #B*embed_dim
                    #select_feature_with_weight[masks_list[b] == 0] = 0
                else:
                    logits = heads[b](self.norm(x)[:, 0])
                    probs = F.softmax(logits, dim=1)
                    select_feature_with_weight = probs @ heads[b].weight

                    #select_feature_with_weight = torch.zeros([B,1,emb_dim]).cuda() #30.084%@13epoch
                '''
                x_norm = F.normalize(self.norm(x)[:, 0])
                h_norm = F.normalize(heads[b].weight)
                ans = x_norm@h_norm.T
                weight_value, weight_index = torch.max(ans, dim=1)
                select_feature_with_weight = (weight_value.reshape(-1,1)) * ((heads[b].weight)[weight_index]) #30.056%@13epoch
                '''

                '''
                logits = heads[b](self.norm(x)[:, 0])
                probs = F.softmax(logits, dim=1)
                weight_value, weight_index = torch.max(probs, dim=1)
                select_feature_with_weight = (weight_value.reshape(-1,1)) * ((heads[b].weight)[weight_index])
                '''
                #print(logits)
                #print(logits.shape)
                #exit()

                #heads[b].weight
                #self.norm(x)



                """
                I am not sure whether there should be detach(). Stay tuned!
                """

                '''
                prompt = torch.cat([torch.zeros([B,1,emb_dim]).cuda(), \
                           select_feature_with_weight.detach().reshape((B,1,emb_dim)), \
                           torch.zeros([B,num_patch,emb_dim]).cuda()], dim=1)
                '''
                #prompt = select_feature_with_weight.detach().reshape((B,1,emb_dim))
                prompt = select_feature_with_weight.detach().reshape((B, 1, emb_dim))
                prompt = torch.nn.functional.pad(prompt, (0, 0, 1+b-8, num_patch+2-(b-8)))
                #print(prompt.shape)
                x = x + prompt
            
            #x[:, 1] = x[:, 1] + select_feature_with_weight.detach() 
            #exit()
            '''
            x[:, 1][masks_list[b] == 1] = x[:, 1][masks_list[b] == 1] + \
            select_feature_with_weight[masks_list[b] == 1].detach()
            '''
        #exit()
        

        #x = self.norm(x)
        return xs[0][:, 0], xs[1][:, 0], xs[2][:, 0], xs[3][:, 0], xs[4][:, 0], xs[5][:, 0], \
               xs[6][:, 0], xs[7][:, 0], xs[8][:, 0], xs[9][:, 0], xs[10][:, 0], xs[11][:, 0]

    def forward(self, x, targets_list):
        x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11 = self.forward_features(x, targets_list)
        
        x_11 = self.head(x_11)
        if not self.training:
            # during inference, return the last classifier predictions
            return x_11
        
        x_0 = self.head_0(x_0)
        x_1 = self.head_1(x_1)
        x_2 = self.head_2(x_2)
        x_3 = self.head_3(x_3)
        x_4 = self.head_4(x_4)
        x_5 = self.head_5(x_5)
        x_6 = self.head_6(x_6)
        x_7 = self.head_7(x_7)
        x_8 = self.head_8(x_8)
        x_9 = self.head_9(x_9)
        x_10 = self.head_10(x_10)
        if self.training:
            return x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11

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