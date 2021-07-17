import torch
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
