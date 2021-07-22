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
        self.tau = torch.nn.Parameter(1)

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
        divider = sqrt(self.embed_size)
        mh_out = torch.softmax(energy_term, -1)
        if self.attention_store is not None:
            self.attention_store.append(mh_out.detach().cpu())
        out = torch.einsum('behq, bnhe -> bnhq ', mh_out / divider, values)
        print(out.shape)
        out = einops.rearrange(out, "b n h e -> b n (h e)")
        return self.final_projection(out)
    

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
    
    def __init__(self):
        super().__init__()


class XCIT(nn.Module):
    
    def __init__(self):
        super(XCIT, self).__init__()