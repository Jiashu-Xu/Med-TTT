# patch_embedding.py

import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.projection(x)  # Shape: [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2)        # Shape: [B, embed_dim, N_patches]
        x = x.transpose(1, 2)   # Shape: [B, N_patches, embed_dim]
        return x
