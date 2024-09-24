# rms_norm.py

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # hidden_states: [B, N, C]
        variance = hidden_states.pow(2).mean(-1, keepdim=True)  # [B, N, 1]
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)  # [B, N, C]
        return self.weight * hidden_states  # [B, N, C]
