from math import sqrt

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.ops import PositionalEncodingFourier, DropPath


class MultiHeadXCITAttention(torch.nn.Module):
    def __init__(self, embed_size, num_heads, attention_dropout_rate, projection_dropout_rate, attention_store=None):
        super().__init__()
        self.queries_projection = nn.Linear(embed_size, embed_size)
        self.values_projection = nn.Linear(embed_size, embed_size)
        self.keys_projection = nn.Linear(embed_size, embed_size)
        self.final_projection = nn.Linear(embed_size, embed_size)
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.attention_store = attention_store
        self.tau = torch.nn.Parameter(torch.ones(embed_size // num_heads))
        self.attention_dropout = nn.Dropout(attention_dropout_rate)
        self.projection_dropout = nn.Dropout(projection_dropout_rate)

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
        mh_out = torch.softmax(energy_term, -1)
        mh_out = self.attention_dropout(mh_out)
        if self.attention_store is not None:
            self.attention_store.append(mh_out.detach().cpu())
        out = torch.einsum('behq, bnhe -> bnhq ', mh_out / self.tau, values)
        out = einops.rearrange(out, "b n h e -> b n (h e)")
        return self.projection_dropout(self.final_projection(out))


class ClassAttention(nn.Module):

    def __init__(self, embed_size, num_heads, attention_dropout_rate, projection_dropout_rate):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.divider = sqrt(self.embed_size // self.num_heads)
        self.projection = nn.Linear(embed_size, embed_size * 3)
        self.output_projection = nn.Linear(embed_size, embed_size)
        self.attention_dropout = nn.Dropout(attention_dropout_rate)
        self.projection_dropout = nn.Dropout(projection_dropout_rate)

    def forward(self, x):
        qkv = einops.rearrange(self.projection(x), "b n (a c) -> a b n c", a=3)
        q, k, v = qkv[0, ...], qkv[1, ...], qkv[2, ...]
        keys = einops.rearrange(k, "b n (h e) -> b n h e", h=self.num_heads)
        queries = einops.rearrange(q, "b n (h e) -> b n h e", h=self.num_heads)
        values = einops.rearrange(v, "b n (h e) -> b h n e", h=self.num_heads)
        queries = queries[:, 0:1, :, :]
        attention = (queries * keys).sum(-1) / self.divider
        attention = einops.rearrange(attention.softmax(1), "b h n -> b n h")
        attention = self.attention_dropout(attention)
        attention = einops.rearrange(attention.unsqueeze(2) @ values, "b h t e -> b t (h e)")
        token = self.output_projection(attention)
        self.projection_dropout(token)
        return torch.cat([token, x[:, 1:, :]], dim=1)


class ClassAttentionLayer(nn.Module):

    def __init__(self, embed_size, num_heads, use_token_norm,
                 attention_dropout_rate=0., projection_dropout_rate=0.,
                 drop_path_rate=0.):
        super(ClassAttentionLayer, self).__init__()
        self.attention = ClassAttention(
            embed_size=embed_size,
            num_heads=num_heads,
            attention_dropout_rate=attention_dropout_rate,
            projection_dropout_rate=projection_dropout_rate
        )
        self.use_token_norm = use_token_norm
        self.mlp = MLP(embed_size=embed_size)
        self.norm_attention = nn.LayerNorm(embed_size)
        self.norm_mlp = nn.LayerNorm(embed_size)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        x = x + self.drop_path(self.attention(self.norm_attention(x)))

        if self.use_token_norm:
            x = self.norm_mlp(x)

        else:
            x[:, 0:1, :] = self.norm_mlp(x[:, 0:1, :])

        cls_token = x[:, 0:1, :]
        cls_token = cls_token + self.mlp(cls_token)
        out_x = torch.cat([cls_token, x[:, 1:, :]], dim=1)
        return x + self.drop_path(out_x)


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


class ResidualAdd(nn.Module):
    def __init__(self, block, drop_path_rate=0.):
        super().__init__()
        self.block = block
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        return self.drop_path(self.block(x)) + x


class XCITLayer(nn.Module):
    def __init__(self, embed_size, num_heads, kernel_size, padding,
                 attention_dropout_rate=0., projection_dropout_rate=0., drop_path_rate=0.):
        super(XCITLayer, self).__init__()
        self.lpi = LPI(in_channels=embed_size, kernel_size=kernel_size, padding=padding)
        self.norm = nn.LayerNorm(embed_size)
        self.mh = ResidualAdd(nn.Sequential(*[
            nn.LayerNorm(embed_size),
            MultiHeadXCITAttention(
                embed_size=embed_size,
                num_heads=num_heads,
                attention_dropout_rate=attention_dropout_rate,
                projection_dropout_rate=projection_dropout_rate
            )
        ]))
        self.mlp = ResidualAdd(nn.Sequential(*[
            nn.LayerNorm(embed_size),
            MLP(embed_size=embed_size)
        ]))
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x, w, h):
        x = self.mh(x)
        x = x + self.drop_path(self.lpi(self.norm(x), w, h))
        return self.mlp(x)


class XCIT(nn.Module):
    
    def __init__(self, num_classes, num_class_attention_layers, num_xcit_layers, patch_size,
                 embed_size=768, num_heads=8, attention_dropout_rate=0., projection_dropout_rate=0.,
                 drop_path_rate=0., eta=0.,
                 use_token_norm=True, kernel_size=3, padding=1, use_pos_encoding=True):
        super(XCIT, self).__init__()
        self._patch_embedding = ConvPatchEmbedding(stride=patch_size, embed_size=embed_size)
        self._pos_embedding = PositionalEncodingFourier(dim=embed_size) if use_pos_encoding else None
        self._xcit_layers = nn.ModuleList([
            XCITLayer(
                embed_size=embed_size,
                num_heads=num_heads,
                kernel_size=kernel_size,
                padding=padding,
                attention_dropout_rate=attention_dropout_rate,
                projection_dropout_rate=projection_dropout_rate,
                drop_path_rate=drop_path_rate
            ) for _ in range(num_xcit_layers)])
        self._class_layers = nn.ModuleList([
            ClassAttentionLayer(
                embed_size=embed_size,
                num_heads=num_heads,
                use_token_norm=use_token_norm,
                attention_dropout_rate=attention_dropout_rate,
                projection_dropout_rate=projection_dropout_rate,
                drop_path_rate=drop_path_rate
            ) for _ in range(num_class_attention_layers)])
        self.head = nn.Linear(embed_size, num_classes)
        self._cls_token = torch.nn.Parameter(torch.ones(1, 1, embed_size))
        self._norm_final = nn.LayerNorm(embed_size)

    def forward(self, image):
        patches, w, h = self._patch_embedding(image)
        if self._pos_embedding is not None:
            pos_embedding = self._pos_embedding(patches.shape[0], w, h)
            pos_embedding = einops.rearrange(pos_embedding, "b e w h -> b (w h) e")
        else:
            pos_embedding = torch.zeros_like(patches, requires_grad=False)
        feats = patches + pos_embedding

        for xcit_layer in self._xcit_layers:
            feats = xcit_layer(feats, w=w, h=h)

        tokenized = torch.cat([einops.repeat(self._cls_token, "b n e -> (repeat b) n e", repeat=feats.shape[0]), feats], dim=1)

        for class_ in self._class_layers:
            tokenized = class_(tokenized)

        token = self._norm_final(tokenized)[:, :1, :]
        return self.head(token)


if __name__ == "__main__":
    from time import time

    model = XCIT(2, 6, 6, 16, 384, use_pos_encoding=True, attention_dropout_rate=0.1,
                 projection_dropout_rate=0.1, drop_path_rate=0.5).cuda()
    N = 100
    start = time()
    for idx in range(N):
        out = model(torch.rand(2, 3, 768, 768).cuda())
    print(f"FPS: {N / (time() - start)}, TIME_PER_ITER: {(time() - start) / N}")
    print(out.shape)