# ttt_cache_2d.py

from collections import defaultdict
from typing import List, Tuple
import torch
import torch.nn as nn

class TTTCache2D:
    """
    TTTCache2D is a data structure that holds the last hidden states and gradients for the TTT layer in 2D.
    """

    def __init__(self, model, batch_size: int, image_size: Tuple[int, int]):
        config = model.config
        self.seqlen_offset = 0
        self.mini_batch_size = config.mini_batch_size
        self.image_size = image_size  # (height, width)

        self.ttt_params_dict = defaultdict(dict)
        if "linear" in config.ttt_layer_type:
            self.ttt_param_names = ["W1", "b1"]
        elif "mlp" in config.ttt_layer_type:
            self.ttt_param_names = ["W1", "b1", "W2", "b2"]
        else:
            raise ValueError(f"TTT Layer Type {config.ttt_layer_type} not supported yet")

        self.conv_states_dict = defaultdict(dict)
        print(f"Creating cache of size: {batch_size}")
        for layer_idx in range(config.num_hidden_layers):
            for name in self.ttt_param_names:
                weight = getattr(model.layers[layer_idx].seq_modeling_block, name)
                tiled_weight = torch.tile(weight.unsqueeze(0), (batch_size,) + (1,) * weight.dim()).to(model.device)
                self.ttt_params_dict[f"{name}_states"][layer_idx] = tiled_weight
                # For decoding, we need to store the gradients as well
                self.ttt_params_dict[f"{name}_grad"][layer_idx] = torch.zeros_like(tiled_weight)

            if config.pre_conv:
                # Adjust conv states for 2D convolution
                self.conv_states_dict["pre_conv"][layer_idx] = torch.zeros(
                    batch_size,
                    config.hidden_size,
                    self.image_size[0],
                    self.image_size[1],
                    device=model.device,
                )
            if config.share_qk:
                self.conv_states_dict["ttt_conv_q"][layer_idx] = torch.zeros(
                    batch_size,
                    config.hidden_size,
                    self.image_size[0],
                    self.image_size[1],
                    device=model.device,
                )
                self.conv_states_dict["ttt_conv_k"][layer_idx] = torch.zeros(
                    batch_size,
                    config.hidden_size,
                    self.image_size[0],
                    self.image_size[1],
                    device=model.device,
                )

    def update(self, py_tree, layer_idx, seq_len):
        # Implement update logic specific to 2D data if necessary
        pass

    def ttt_params_to_dict(self, layer_idx):
        return {name: self.ttt_params_dict[name][layer_idx] for name in self.ttt_params_dict}
