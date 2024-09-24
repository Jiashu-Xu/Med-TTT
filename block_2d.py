# block_2d.py

from ttt_mlp_2d import TTTMLP2D
from rms_norm import RMSNorm
from swi_glu_mlp import SwiGluMLP
import torch.nn as nn

class Block2D(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.pre_conv = config.pre_conv

        if config.ttt_layer_type == "mlp":
            ttt_layer = TTTMLP2D
        else:
            raise ValueError(f"Invalid ttt_layer_type: {config.ttt_layer_type}")

        self.seq_modeling_block = ttt_layer(config=config, layer_idx=layer_idx)

        self.mlp = SwiGluMLP(config)

        if self.pre_conv:
            self.conv = nn.Conv2d(
                self.hidden_size,
                self.hidden_size,
                kernel_size=config.conv_kernel,
                padding=config.conv_kernel // 2,
                groups=self.hidden_size,
            )

        self.seq_norm = RMSNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        cache_params=None,
    ):
        if self.pre_conv:
            residual = hidden_states
            B, N, C = hidden_states.shape
            H = W = int(N ** 0.5)
            hidden_states_reshaped = hidden_states.transpose(1, 2).view(B, C, H, W)
            hidden_states_conv = self.conv(hidden_states_reshaped)
            hidden_states_conv = hidden_states_conv.view(B, C, N).transpose(1, 2)
            hidden_states = residual + hidden_states_conv

        residual = hidden_states
        hidden_states = self.seq_norm(hidden_states)

        # TTT Layer
        hidden_states, _ = self.seq_modeling_block(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_params=cache_params,
        )
        hidden_states = residual + hidden_states

        # Feed-Forward Network
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        #print("hidden_states",hidden_states.shape)
        return hidden_states
