import itertools
from math import sqrt
import einops
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import GELU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d


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
        self.tau = torch.nn.Parameter(num_heads)

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
class ClassAttentionLayer(nn.Module):

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



class Conv3x3(nn.Sequential):
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(*[
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride),
            nn.BatchNorm2d(out_channels)
        ])


class ConvPatchEmbedding(nn.Module):
    
    def __init__(self, stride=16, embed_dim=768):
        super().__init__()
        num_conv_layers = int(torch.log2(torch.tensor(stride)))
        self.patch_embedding = self._get_patch_embedding(num_conv_layers, stride, embed_dim=embed_dim)
        
    def _get_patch_embedding(self, num_conv_layers, stride, embed_dim):
        embedding = [Conv3x3(in_channels=3, out_channels=embed_dim // (stride // 2), stride=2), GELU()]
        for idx in range(1, num_conv_layers - 1):
            embedding += [Conv3x3(in_channels=embed_dim // (2 ** idx), out_channels=embed_dim // (2 ** (idx + 1)), stride=2),  GELU()]
        embedding += [Conv3x3(in_channels=embed_dim // 2, out_channels=embed_dim, stride=2), GELU()]
        return nn.ModuleList(embedding)
        
    def forward(self, image):
        embed = self.patch_embedding(image)
        _, _, w, h = embed.shape
        return einops.rearrange(embed, "b c w h -> b (w h) c"), (w, h)


class LPI(nn.Sequential):
    
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
        x = einops.rearrange(x, "b n c -> b c h w", h=h, w=w)
        lpi = self.lpi(x)
        return einops.rearrange(lpi, "b c h w -> b n c")


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


class XCIT(nn.Module):
    
    def __init__(self):
        super(XCIT, self).__init__()


if __name__ == "__main__":
    ClassAttentionLayer(embed_size=768, num_heads=8)(torch.rand(2, 197, 768))