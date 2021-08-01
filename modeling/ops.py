from collections import OrderedDict
import math
from typing import List

import torch
import torch.nn as nn
from torch import nn as nn


def apply_ema(teacher: torch.nn.Module, student: torch.nn.Module, decay: float) -> torch.nn.Module:
    t_dict = teacher.state_dict()
    s_dict = student.state_dict()
    t_dict_new = OrderedDict()
    for name, params in s_dict.items():
        if name in t_dict:
            t_dict_new[name] = decay * t_dict[name] + (1 - decay) * params

    teacher.load_state_dict(t_dict_new)
    return teacher


class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W):
        mask = torch.zeros(B, H, W).bool().to(self.token_projection.weight.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos


class DropPath(nn.Module):

    def __init__(self, p=0.0):
        super().__init__()
        assert 0. <=p <= 1.
        self.p = p

    def forward(self, input_vector):
        if not self.training or self.p == 0.0:
            return input_vector
        drop_mask = (torch.rand(input_vector.shape, device=input_vector.device) > self.p).long()
        return torch.div(input_vector, 1. - self.p) * drop_mask


class Sequential(nn.Module):

    def __init__(self, blocks: List[nn.Module]):
        super(Sequential, self).__init__()
        self.blocks = blocks

    def forward(self, x, *args, **kwargs):
        for block in self.blocks:
            x = block(x, *args, **kwargs)
        return x