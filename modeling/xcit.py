from math import sqrt
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


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



class XCIT(nn.Module):
    
    def __init__(self):
        super(XCIT, self).__init__()