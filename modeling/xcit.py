import itertools
from math import sqrt
from typing import List

import einops
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import GELU


class MultiHeadXCITAttention(torch.nn.Module):
    def __init__(self, embed_size, num_heads, attention_store=None):
        super().__init__()
        self.queries_projection = nn.Linear(embed_size, embed_size)
        self.values_projection = nn.Linear(embed_size, embed_size)
        self.keys_projection = nn.Linear(embed_size, embed_size)
        self.final_projection = nn.Linear(embed_size, embed_size)
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.attention_store = attention_store
        self.tau = torch.nn.Parameter(torch.ones(embed_size // num_heads))

    def forward(self, x):
        assert len(x.shape) == 3
        keys = self.keys_projection(x)
        values = self.values_projection(x)
        queries = self.queries_projection(x)
        keys = einops.rearrange(keys, "b n (h e) -> b n h e", h=self.num_heads)
        queries = einops.rearrange(queries, "b n (h e) -> b n h e", h=self.num_heads)
        values = einops.rearrange(values, "b n (h e) -> b n h e", h=self.num_heads)
        keys = F.normalize(keys, p=2, dim=1)
        queries = F.normalize(queries, p=2, dim=1)
        energy_term = torch.einsum("bnhe, bnhq -> behq", queries, keys)
        print(energy_term.shape)
        mh_out = torch.softmax(energy_term, -1)
        if self.attention_store is not None:
            self.attention_store.append(mh_out.detach().cpu())
        out = torch.einsum('behq, bnhe -> bnhq ', mh_out / self.tau, values)
        print(out.shape)
        out = einops.rearrange(out, "b n h e -> b n (h e)")
        return self.final_projection(out)


# TODO: add dropout
class ClassAttention(nn.Module):

    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.divider = sqrt(self.embed_size // self.num_heads)
        self.projection = nn.Linear(embed_size, embed_size * 3)
        self.output_projection = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        qkv = einops.rearrange(self.projection(x), "b n (a c) -> a b n c", a=3)
        q, k, v = qkv[0, ...], qkv[1, ...], qkv[2, ...]
        keys = einops.rearrange(k, "b n (h e) -> b n h e", h=self.num_heads)
        queries = einops.rearrange(q, "b n (h e) -> b n h e", h=self.num_heads)
        values = einops.rearrange(v, "b n (h e) -> b h n e", h=self.num_heads)
        queries = queries[:, 0:1, :, :]
        attention = (queries * keys).sum(-1) / self.divider
        attention = einops.rearrange(attention.softmax(1), "b h n -> b n h")
        attention = einops.rearrange(attention.unsqueeze(2) @ values, "b h t e -> b t (h e)")
        token = self.output_projection(attention)
        return torch.cat([token, x[:, 1:, :]], dim=1)


class ClassAttentionLayer(nn.Module):

    def __init__(self, embed_size, num_heads, use_token_norm):
        super(ClassAttentionLayer, self).__init__()
        self.attention = ClassAttention(embed_size=embed_size, num_heads=num_heads)
        self.use_token_norm = use_token_norm
        self.mlp = MLP(embed_size=embed_size)
        self.norm_attention = nn.LayerNorm(embed_size)
        self.norm_mlp = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.attention(self.norm_attention(x))

        if self.use_token_norm:
            x = self.norm_mlp(x)

        else:
            x[:, 0:1, :] = self.norm_mlp(x[:, 0:1, :])

        cls_token = x[:, 0:1, :]
        cls_token = cls_token + self.mlp(cls_token)
        out_x = torch.cat([cls_token, x[:, 1:, :]], dim=1)
        return x + out_x


class Conv3x3(nn.Sequential):
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(*[
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride),
            nn.BatchNorm2d(out_channels)
        ])


class ConvPatchEmbedding(nn.Module):

    def __init__(self, stride=16, embed_size=768):
        super().__init__()
        num_conv_layers = int(torch.log2(torch.tensor(stride)))
        self.patch_embedding = self._get_patch_embedding(num_conv_layers, stride, embed_dim=embed_size)
        print(self)

    def _get_patch_embedding(self, num_conv_layers, stride, embed_dim):
        embedding = [Conv3x3(in_channels=3, out_channels=embed_dim // (stride // 2), stride=2), nn.GELU()]
        for idx in range(num_conv_layers - 1, 1, -1):
            embedding += [
                Conv3x3(in_channels=embed_dim // (2 ** idx), out_channels=embed_dim // (2 ** (idx - 1)), stride=2),
                nn.GELU()]
        embedding += [Conv3x3(in_channels=embed_dim // 2, out_channels=embed_dim, stride=2)]
        return nn.Sequential(*embedding)

    def forward(self, image):
        embed = self.patch_embedding(image)
        _, _, w, h = embed.shape
        return einops.rearrange(embed, "b c w h -> b (w h) c"), w, h


class LPI(nn.Module):
    
    def __init__(self, in_channels, kernel_size=3, padding=1):
        super().__init__()
        self.lpi = nn.Sequential(*[
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        ])
    
    def forward(self, x, w, h):
        x = einops.rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        lpi = self.lpi(x)
        return einops.rearrange(lpi, "b c h w -> b (h w) c")


class MLP(torch.nn.Sequential):
    def __init__(self, embed_size=768, expansion=4):
        super().__init__(*[
            nn.Linear(embed_size, embed_size * expansion),
            nn.GELU(),
            nn.Linear(embed_size * expansion, embed_size)
        ])
        

class DropPath(nn.Module):
    
    def __init__(self, p=0.0):
        super().__init__()
        assert 0. <=p <= 1.
        self.p = p

    def forward(self, input_vector):
        if not self.training or self.p == 0.0:
            return input_vector
        drop_mask = (torch.rand(input_vector.shape) > self.p).long()
        return torch.div(input_vector, 1. - self.p) * drop_mask


class ResidualAdd(nn.Module):
    def __init__(self, block):
        super(ResidualAdd, self).__init__()
        self.block = block

    def forward(self, x, *args, **kwargs):
        return self.block(x, *args, **kwargs) + x


class Sequential(nn.Module):

    def __init__(self, blocks: List[nn.Module]):
        super(Sequential, self).__init__()
        self.blocks = blocks

    def forward(self, x, *args, **kwargs):
        for block in self.blocks:
            x = block(x, *args, **kwargs)
        return x


class XCITLayer(nn.Module):
    def __init__(self, embed_size, num_heads, kernel_size, padding):
        super(XCITLayer, self).__init__()
        self.lpi = LPI(in_channels=embed_size, kernel_size=kernel_size, padding=padding)
        self.norm = nn.LayerNorm(embed_size)
        self.mh = ResidualAdd(nn.Sequential(*[
            nn.LayerNorm(embed_size),
            MultiHeadXCITAttention(embed_size=embed_size, num_heads=num_heads)
        ]))
        self.mlp = ResidualAdd(nn.Sequential(*[
            nn.LayerNorm(embed_size),
            MLP(embed_size=embed_size)
        ]))

    def forward(self, x, w, h):
        x = self.mh(x)
        x = x + self.lpi(self.norm(x), w, h)
        return self.mlp(x)


class XCIT(nn.Module):
    
    def __init__(self, num_classes, num_class_attention_layers, num_xcit_layers, patch_size, embed_size=768, num_heads=8, use_dropout=False, use_token_norm=True, kernel_size=3, padding=1, use_pos_encoding=True):
        super(XCIT, self).__init__()
        self._patch_embedding = ConvPatchEmbedding(stride=patch_size, embed_size=embed_size)
        self._pos_embedding = None if use_pos_encoding else nn.Identity()
        self._blck_layers = [
            XCITLayer(
                embed_size=embed_size,
                num_heads=num_heads,
                kernel_size=kernel_size,
                padding=padding
            ) for _ in range(num_xcit_layers)]
        self._cls_layers = [
            ClassAttentionLayer(
                embed_size=embed_size,
                num_heads=num_heads,
                use_token_norm=use_token_norm
            ) for _ in range(num_class_attention_layers)]
        self.head = nn.Linear(embed_size, num_classes)
        self._cls_token = torch.nn.Parameter(torch.ones(1, 1, embed_size))
        self._norm_final = nn.LayerNorm(embed_size)

    def forward(self, image):
        patches, w, h = self._patch_embedding(image)
        feats = self._pos_embedding(patches)
        for xcit in self._blck_layers:
            feats = xcit(feats, w=w, h=h)

        tokenized = torch.cat([einops.repeat(self._cls_token, "b n e -> (repeat b) n e", repeat=feats.shape[0]), feats], dim=1)

        for class_ in self._cls_layers:
            tokenized = class_(tokenized)

        token = self._norm_final(tokenized)[:, :1, :]
        return self.head(token)


if __name__ == "__main__":
    out = XCIT(2, 6, 6, 16, 768, use_pos_encoding=False)(torch.rand(2, 3, 256, 256))
    print(out.shape)