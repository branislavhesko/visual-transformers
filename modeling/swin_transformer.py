from math import sqrt

import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange, Reduce


class MLP(torch.nn.Sequential):
    def __init__(self, embed_size=768, expansion=4):
        super().__init__(*[
            nn.Linear(embed_size, embed_size * expansion),
            nn.GELU(),
            nn.Linear(embed_size * expansion, embed_size)
        ])


class PositionalEncodingSinusoidal(torch.nn.Module):
    pass


class PatchEmbeddingPixelwise(torch.nn.Sequential):
    
    def __init__(self, stride, embedding_size, channels=3) -> None:
        reduce = Reduce("b c (w i) (h k) -> b (c i k) w h", "mean", i=stride, k=stride)
        rearange = Rearrange("b e h w -> b (h w) e")
        linear = torch.nn.Linear(stride * stride * channels, embedding_size)
        super().__init__(*[
            reduce,
            rearange,
            linear
        ])


class MLP(torch.nn.Sequential):
    def __init__(self, embed_size=768, expansion=4):
        super().__init__(*[
            nn.Linear(embed_size, embed_size * expansion),
            nn.GELU(),
            nn.Linear(embed_size * expansion, embed_size)
        ])


class ResidualAdd(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)
    

class SwinMSA(nn.Module):
    
    def __init__(self, embed_dim, num_heads, attention_mask=None,
                 attention_dropout=0.0, projection_dropout=0.0):
        super(SwinMSA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.scale = (self.embed_dim // self.num_heads) ** (-0.5)

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection_dropout = nn.Dropout(projection_dropout)

        self.register_buffer("attention_mask", attention_mask)

    def forward(self, x):
        """
        Multihead self-attention for swin module.
        Args:
            x: tensor of shape: b n c

        Returns:

        """
        print(x.shape)
        qkv = self.qkv(x)
        qkv = einops.rearrange(qkv, "b n (h d c) -> d b n h c", h=self.num_heads, d=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy_term = torch.einsum("bqhe, bkhe -> bqhk", queries, keys)
        energy_term *= self.scale

        if self.attention_mask is not None:
            energy_term = energy_term + self.attention_mask

        energy_term = energy_term.softmax(dim=-1)
        out = torch.einsum('bihv, bvhd -> bihd ', energy_term, values)
        print(out.shape)
        out = einops.rearrange(out, "b n h e -> b n (h e)")
        out = self.projection(out)
        return self.projection_dropout(out)


class SwinBlock(nn.Module):
    
    def __init__(self, shift_size, embed_dim, num_heads, image_resolution,
                 shifted_block=False, attention_dropout=0.0, projection_dropout=0.0):
        super(SwinBlock, self).__init__()
        self.resolution = image_resolution
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.shifted = shifted_block
        self.shift_size = shift_size
        self.attention_norm = nn.LayerNorm(embed_dim)
        attention_mask = self._get_attention_mask()

        self.attention = SwinMSA(
            embed_dim,
            num_heads,
            attention_mask=attention_mask,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout
        )
        self.mlp = ResidualAdd(nn.Sequential(*[nn.LayerNorm(embed_dim), MLP(embed_size=embed_dim)]))
        
    def _get_attention_mask(self):
        if self.shifted:
            # TODO: fill
            return torch.ones(1)
        else:
            return None

    def cyclic_shift(self, tensor):
        return torch.roll(tensor, (-self.shift_size, -self.shift_size), dims=(1, 2))

    def reverse_cyclic_shift(self, tensor):
        return torch.roll(tensor, (self.shift_size, self.shift_size), dims=(1, 2))

    def window_reverse(self, tensor, h , w):
        image = einops.rearrange(tensor, "(b h w) (shifty shiftx) c -> b shifty h shiftx w c",
                                 shiftx=self.shift_size, shifty=self.shift_size, h=h//self.shift_size, w=w//self.shift_size)
        return einops.rearrange(image, "b sy h sx w c -> b (sy h) (sx w) c")

    def window_partition(self, tensor):
        windows = einops.rearrange(tensor, "b (sy h) (sx w) c -> b sy h sx w c", sx=self.shift_size, sy=self.shift_size)
        return einops.rearrange(windows, "b sy h sx w c -> (b h w) (sy sx) c")

    def forward(self, x):
        """
        Single swin block execution
        Args:
            img: (B (H W) C) tensor.

        Returns:

        """
        b, n, c = x.shape
        assert n == self.resolution[0] * self.resolution[1]
        img = einops.rearrange(x, "b (h w) c ->  b h w c", h=self.resolution[0], w=self.resolution[1])
        if self.shifted:
            img = self.cyclic_shift(img)
        print(img.shape)
        norm1 = self.attention_norm(img)
        print(norm1.shape)
        partitions = self.window_partition(norm1)
        print(partitions.shape)
        attention = self.attention(partitions)
        reverse_shifted = self.window_reverse(attention, *self.resolution)
        if self.shifted:
            reverse_shifted = self.reverse_cyclic_shift(reverse_shifted)
        msa_out = einops.rearrange(reverse_shifted, "b h w c -> b (h w) c")
        msa_out = x + msa_out
        print(msa_out.shape)
        return msa_out + self.mlp(msa_out)
    

class SwinTransformer(nn.Module):
    pass
