import torch
import torch.nn as nn

class FeedForward(nn.Module):

    def __init__(self, d_model, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim),
            nn.ReLU(),
            nn.Linear(dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    