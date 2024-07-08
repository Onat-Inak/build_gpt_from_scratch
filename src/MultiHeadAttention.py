import torch
import torch.nn as nn
from Head import Head as Head

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, d_model, d_k, d_v, n, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(d_model, d_k, d_v, n) for _ in range(num_heads)])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, emb, bool_mask = True):
        out = torch.cat([h(emb, bool_mask) for h in self.heads], dim = -1)
        return self.dropout(self.proj(out))