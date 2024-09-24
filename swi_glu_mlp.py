# swi_glu_mlp.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGluMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = getattr(F, config.hidden_act) if hasattr(F, config.hidden_act) else nn.SiLU()

    def forward(self, x):
        gate = self.gate_proj(x)  # [B, N, intermediate_size]
        up = self.up_proj(x)      # [B, N, intermediate_size]
        intermediate_states = self.act_fn(gate) * up  # [B, N, intermediate_size]
        output = self.down_proj(intermediate_states)  # [B, N, hidden_size]
        return output
