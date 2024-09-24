# positional_embedding.py

import torch
import torch.nn as nn

class Learned2DPositionalEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.row_embed = nn.Parameter(torch.randn(int(num_patches ** 0.5), embed_dim // 2))
        self.col_embed = nn.Parameter(torch.randn(int(num_patches ** 0.5), embed_dim // 2))

    def forward(self, x):
        B, N, D = x.size()
        H = W = int(N ** 0.5)
        row_embed = self.row_embed[:H].unsqueeze(1).repeat(1, W, 1)
        col_embed = self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1)
        pos_embed = torch.cat([row_embed, col_embed], dim=-1).view(N, D)
        pos_embed = pos_embed.unsqueeze(0).repeat(B, 1, 1)
        return pos_embed
