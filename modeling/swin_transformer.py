import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange, Reduce


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

    def __init__(self, embed_dim, num_heads, window_size, attention_mask=None,
                 attention_dropout=0.0, projection_dropout=0.0):
        super(SwinMSA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.scale = (self.embed_dim // self.num_heads) ** (-0.5)

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection_dropout = nn.Dropout(projection_dropout)
        self.relative_position_bias_table = torch.nn.Parameter(torch.zeros(
            (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        self.register_buffer("attention_mask", attention_mask)
        self.register_buffer("relative_bias_index", self._get_relative_bias_index(window_size=window_size))

    @staticmethod
    def _get_relative_bias_index(window_size):
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords_new = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords_new.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_bias_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_bias_index

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
        relative_position_bias = self.relative_position_bias_table[self.relative_bias_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        energy_term = energy_term + relative_position_bias
        if self.attention_mask is not None:
            number_of_windows = self.attention_mask.shape[0]
            energy_term = einops.rearrange(energy_term, "(b nw) h n n -> b nw h n n", nw=number_of_windows)
            energy_term = energy_term + self.attention_mask.unsqueeze(1).unsqueeze(0)
            energy_term = einops.rearrange(energy_term, "b nw h n n -> (b nw) h n n")
        energy_term = energy_term.softmax(dim=-1)
        out = torch.einsum('bihv, bvhd -> bihd ', energy_term, values)
        print(out.shape)
        out = einops.rearrange(out, "b n h e -> b n (h e)")
        out = self.projection(out)
        return self.projection_dropout(out)


class SwinBlock(nn.Module):

    def __init__(self, window_size, embed_dim, num_heads, image_resolution, shift_size=0,
                 attention_dropout=0.0, projection_dropout=0.0):
        super(SwinBlock, self).__init__()
        self.window_size = window_size
        self.resolution = image_resolution
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.attention_norm = nn.LayerNorm(embed_dim)
        attention_mask = self._get_attention_mask(shift_size, window_size, image_resolution[1], image_resolution[0])
        self.attention = SwinMSA(
            embed_dim,
            num_heads,
            window_size=window_size,
            attention_mask=attention_mask,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout
        )
        self.mlp = ResidualAdd(nn.Sequential(*[nn.LayerNorm(embed_dim), MLP(embed_size=embed_dim)]))

    def _get_attention_mask(self, shift, window_size, h, w):
        if self.shift_size > 0:
            image_mask = torch.zeros((1, h, w, 1))
            h_slices = (slice(0, -window_size),
                        slice(-window_size, -shift),
                        slice(-shift, None))
            w_slices = (slice(0, -window_size),
                        slice(-window_size, -shift),
                        slice(-shift, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    image_mask[:, h, w, :] = cnt
                    cnt += 1
            attention_mask = self.window_partition(image_mask)
            attention_mask = einops.rearrange(attention_mask, "num_windows (window_size_y window_size_x) channels -> "
                                                              "(num_windows channels) (window_size_y window_size_x)")
            attention_mask = attention_mask.unsqueeze(1) - attention_mask.unsqueeze(2)
            attention_mask = attention_mask.masked_fill(attention_mask != 0, -100.).masked_fill(attention_mask == 0, 0.)
            return attention_mask
        else:
            return None

    def cyclic_shift(self, tensor):
        return torch.roll(tensor, (-self.shift_size, -self.shift_size), dims=(1, 2))

    def reverse_cyclic_shift(self, tensor):
        return torch.roll(tensor, (self.shift_size, self.shift_size), dims=(1, 2))

    def window_reverse(self, tensor, h, w):
        image = einops.rearrange(tensor, "(b h w) (shifty shiftx) c -> b shifty h shiftx w c",
                                 shiftx=self.window_size, shifty=self.window_size,
                                 h=h // self.window_size, w=w // self.window_size)
        return einops.rearrange(image, "b sy h sx w c -> b (sy h) (sx w) c")

    def window_partition(self, tensor):
        windows = einops.rearrange(tensor, "b (wy h) (wx w) c -> b wy h wx w c",
                                   wx=self.window_size, wy=self.window_size)
        return einops.rearrange(windows, "b wy h wx w c -> (b h w) (wy wx) c")

    def forward(self, x):
        """
        Single swin block execution
        Args:
            x: (B (H W) C) tensor.

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
