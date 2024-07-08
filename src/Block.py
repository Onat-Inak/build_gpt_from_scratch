import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention as MultiHeadAttention
from FeedForward import FeedForward as FeedForward

class Block(nn.Module):

    def __init__(self, num_heads, d_model, d_k, d_v, n, mlp_dim, dropout):
        super().__init__()

        self.sa_head = MultiHeadAttention(num_heads, d_model, d_k, d_v, n, dropout)
        self.mlp = FeedForward(d_model, mlp_dim, dropout)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x), bool_mask = True)
        x = x + self.mlp(self.ln2(x))
        return x
