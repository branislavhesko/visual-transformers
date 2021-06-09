from math import sqrt

import einops
from einops.layers.torch import Rearrange, Reduce
import torch
import torch.nn as nn


# TODO: this may be refactored
class PatchEmbedding(torch.nn.Module):
    def __init__(self, in_channels, embed_size, patch_size=16):
        super().__init__()

        # noinspection PyTypeChecker
        self._embedding = nn.Sequential(*[
            nn.Conv2d(in_channels, embed_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        ])

        self._cls_token = torch.nn.Parameter(torch.randn(1, embed_size))

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
    def __init__(self, embed_size, num_heads, attention_store=None):
        super().__init__()
        self.queries_projection = nn.Linear(embed_size, embed_size)
        self.values_projection = nn.Linear(embed_size, embed_size)
        self.keys_projection = nn.Linear(embed_size, embed_size)
        self.final_projection = nn.Linear(embed_size, embed_size)
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.attention_store = attention_store

    def forward(self, x):
        assert len(x.shape) == 3
        keys = self.keys_projection(x)
        values = self.values_projection(x)
        queries = self.queries_projection(x)
        keys = einops.rearrange(keys, "b n (h e) -> b n h e", h=self.num_heads)
        queries = einops.rearrange(queries, "b n (h e) -> b n h e", h=self.num_heads)
        values = einops.rearrange(values, "b n (h e) -> b n h e", h=self.num_heads)
        energy_term = torch.einsum("bqhe, bkhe -> bqhk", queries, keys)
        divider = sqrt(self.embed_size)
        mh_out = torch.softmax(energy_term, -1)
        if self.attention_store is not None:
            self.attention_store.append(mh_out.detach().cpu())
        out = torch.einsum('bihv, bvhd -> bihd ', mh_out / divider, values)
        out = einops.rearrange(out, "b n h e -> b n (h e)")
        return self.final_projection(out)


class MLP(torch.nn.Sequential):
    def __init__(self, embed_size=768, expansion=4):
        super().__init__(*[
            nn.Linear(embed_size, embed_size * expansion),
            nn.GELU(),
            nn.Linear(embed_size * expansion, embed_size)
        ])


class TransformerEncoderLayer(torch.nn.Sequential):
    def __init__(self, embed_size=768, expansion=4, num_heads=8, attention_store=None):
        super(TransformerEncoderLayer, self).__init__(
            *[
                ResidualAdd(nn.Sequential(*[
                    nn.LayerNorm(embed_size),
                    MultiHeadAttention(embed_size, num_heads, attention_store=attention_store)
                ])),
                ResidualAdd(nn.Sequential(*[
                    nn.LayerNorm(embed_size),
                    MLP(embed_size, expansion)
                ]))
            ]
        )


class TransformerEncoder(torch.nn.Sequential):
    def __init__(self, num_layers=6, **kwargs):
        super(TransformerEncoder, self).__init__(*[
            TransformerEncoderLayer(**kwargs) for _ in range(num_layers)
        ])


class ClassificationHead(torch.nn.Sequential):
    def __init__(self, embed_size, num_classes):
        super().__init__(*[
            Reduce("b n e-> b e", reduction="mean"),
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, num_classes)
        ])


class VIT(torch.nn.Module):
    def __init__(self, in_channels, embed_size, num_classes, num_layers,
                 num_heads, image_shape, patch_size, store_attention):
        super().__init__()
        self.attention_store = [] if store_attention else None
        self.patch_embed = PatchEmbedding(in_channels=in_channels, embed_size=embed_size, patch_size=patch_size     )
        self.position_embed = PositionEmbedding(embed_size=embed_size, image_shape=image_shape, patch_size=patch_size)
        self.encoder = TransformerEncoder(num_layers=num_layers, embed_size=embed_size,
                                          num_heads=num_heads, attention_store=self.attention_store)
        self.classifier = ClassificationHead(embed_size=embed_size, num_classes=num_classes)
        self.store_attention = store_attention

    def forward(self, x):
        patches = self.patch_embed(x)
        positions = self.position_embed(patches)
        encoder = self.encoder(positions)
        return self.classifier(encoder)

    def reset(self):
        if self.attention_store is not None and len(self.attention_store) > 0:
            [self.attention_store.pop(0) for _ in range(len(self.attention_store))]


if __name__ == "__main__":
    model = VIT(3, 768, 2, 6, 8, 224, 16, store_attention=False)
    out = model(torch.rand(2, 3, 224, 224))
    print(out)
