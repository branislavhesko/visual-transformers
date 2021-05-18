from math import sqrt

import einops
from einops.layers.torch import Rearrange, Reduce
import torch
import torch.nn as nn

from modeling.vit import TransformerEncoder


class PatchEmbedding(torch.nn.Module):
    def __init__(self, in_channels, embed_size, patch_size=16):
        super().__init__()

        # noinspection PyTypeChecker
        self._embedding = nn.Sequential(*[
            nn.Conv2d(in_channels, embed_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        ])

        self._cls_token = torch.nn.Parameter(torch.randn(1, embed_size))
        self._dis_token = torch.nn.Parameter(torch.randn(1, embed_size))

    def forward(self, x):
        embedded = self._embedding(x)
        cls_token = einops.repeat(self._cls_token, "n e -> b n e", b=x.shape[0])
        dis_token = einops.repeat(self._cls_token, "n e -> b n e", b=x.shape[0])
        return torch.cat([cls_token, dis_token, embedded], dim=1)


class PositionEmbedding(torch.nn.Module):
    def __init__(self, image_shape, patch_size, embed_size):
        super().__init__()
        self._position = nn.Parameter(torch.randn(int((image_shape // patch_size) ** 2 + 2), embed_size))

    def forward(self, x):
        position = einops.repeat(self._position, "n e -> b n e", b=x.shape[0])
        return x + position


class ClassificationHead(torch.nn.Sequential):
    def __init__(self, embed_size, num_classes):
        super(ClassificationHead, self).__init__()
        self.cls_head = nn.Linear(embed_size, num_classes)
        self.dis_head = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        cls_token = x[:, 0, :]
        dis_token = x[:, 1, :]

        cls_out = self.cls_head(cls_token)
        dis_out = self.dis_head(dis_token)
        if self.training:
            return cls_out, dis_out
        return (cls_out + dis_out) / 2.


class DeIT(torch.nn.Module):
    def __init__(self, in_channels, embed_size, num_classes, num_layers, num_heads, image_shape, patch_size):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels=in_channels, embed_size=embed_size)
        self.position_embed = PositionEmbedding(embed_size=embed_size, image_shape=image_shape, patch_size=patch_size)
        self.encoder = TransformerEncoder(num_layers=num_layers, embed_size=embed_size,
                                          num_heads=num_heads)
        self.classifier = ClassificationHead(embed_size=embed_size, num_classes=num_classes)

    def forward(self, x):
        patches = self.patch_embed(x)
        positions = self.position_embed(patches)
        encoder = self.encoder(positions)
        return self.classifier(encoder)


if __name__ == "__main__":
    model = DeIT(3, 768, 2, 6, 8, 224, 16)
    print(model(torch.rand(2, 3, 224, 224)))
