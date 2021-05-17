import einops
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn


class PatchEmbedding(torch.nn.Module):
    def __init__(self, in_channels, num_embeddings, patch_size=16):
        super().__init__()

        # noinspection PyTypeChecker
        self._embedding = nn.Sequential(*[
            nn.Conv2d(in_channels, num_embeddings, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        ])

        self._cls_token = torch.nn.Parameter(torch.randn(1, num_embeddings))

    def forward(self, x):
        embedded = self._embedding(x)
        token = einops.repeat(self._cls_token, "n e -> b n e", b=x.shape[0])
        return torch.cat([token, embedded], dim=1)


class PositionEmbedding(torch.nn.Module):
    def __init__(self, image_shape, patch_size, embed_size):
        super().__init__()
        self._position = nn.Parameter(torch.randn(int((image_shape // patch_size) ** 2 + 1), embed_size))

    def forward(self, x):
        position = einops.repeat(self._position, "n e -> b n e", b=x.shape[0])
        return x + position


class ResidualAdd(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()


class TransformerEncoderLayer(torch.nn.Module):
    pass


class ClassificationHead(torch.nn.Module):
    def __init__(self):
        super().__init__()


class DeIT(torch.nn.Module):
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    embed = PatchEmbedding(3, 768)
    print(embed(torch.rand(1, 3, 128, 128)))