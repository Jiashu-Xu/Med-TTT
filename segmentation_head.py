# segmentation_head.py

import torch.nn as nn

class SegmentationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, hidden_states):
        B, N, D = hidden_states.size()
        H = W = int(N ** 0.5)
        #print("H",H,"w",W)
        x = hidden_states.transpose(1, 2).view(B, D, H, W)
        x = self.conv(x)
        x = nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        return x
