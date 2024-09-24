# ttt_mlp_2d.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from ttt_base_2d import TTTBase2D
from utils import ln_fwd, ln_fused_l2_bwd, gelu_bwd, scan, tree_map

class TTTMLP2D(TTTBase2D):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        # TTT model initialization for TTT-MLP
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, 4 * self.head_dim))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
        self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))
        self.use_dual_form = True  # Set to True as per your original code

        # Define output projection
        self.o_proj = nn.Linear(self.width, self.width, bias=False)


    def ttt(
    self,
    inputs,
    mini_batch_size,
    last_mini_batch_params_dict,
    cache_params=None,
    ):
        B = inputs["XV"].shape[0]
        #print("b",B)
        num_heads = inputs["XV"].shape[1]
        num_mini_batch = inputs["XV"].shape[2]
        K = inputs["XV"].shape[3]
        head_dim = self.head_dim
        #print("k",K,"num_mini_batch",num_mini_batch,"head_dim",head_dim)
        device = inputs["XV"].device
        dtype = inputs["XV"].dtype

        # Define L as total sequence length
        L = num_mini_batch * mini_batch_size

        # In this case, we are decoding
        if last_mini_batch_params_dict is None and cache_params is not None:
            last_mini_batch_params_dict = cache_params.ttt_params_to_dict(self.layer_idx)

        # Initialize parameters
        if last_mini_batch_params_dict is not None:
            init_params_dict = last_mini_batch_params_dict
        else:
            init_params_dict = {
                "W1_states": self.W1.unsqueeze(0).expand(B, -1, -1, -1).clone(),  # [B, num_heads, head_dim, 4*head_dim]
                "b1_states": self.b1.unsqueeze(0).expand(B, -1, -1, -1).clone(),  # [B, num_heads, 1, 4*head_dim]
                "W2_states": self.W2.unsqueeze(0).expand(B, -1, -1, -1).clone(),  # [B, num_heads, 4*head_dim, head_dim]
                "b2_states": self.b2.unsqueeze(0).expand(B, -1, 1, -1).clone(),   # [B, num_heads, 1, head_dim]
            }
            init_params_dict.update(W1_grad=torch.zeros_like(init_params_dict["W1_states"]))
            init_params_dict.update(b1_grad=torch.zeros_like(init_params_dict["b1_states"]))
            init_params_dict.update(W2_grad=torch.zeros_like(init_params_dict["W2_states"]))
            init_params_dict.update(b2_grad=torch.zeros_like(init_params_dict["b2_states"]))

        # Define the compute_mini_batch function
        def compute_mini_batch(params_dict, x):
            # Extract parameters
            W1_init = params_dict["W1_states"]  # [B, num_heads, head_dim, 4*head_dim]
            b1_init = params_dict["b1_states"]  # [B, num_heads, 1, 4*head_dim]
            W2_init = params_dict["W2_states"]  # [B, num_heads, 4*head_dim, head_dim]
            b2_init = params_dict["b2_states"]  # [B, num_heads, 1, head_dim]
    
            # Extract minibatch inputs
            XQ_mini_batch = x["XQ"]  # [B, num_heads, K, head_dim]
            XK_mini_batch = x["XK"]  # [B, num_heads, K, head_dim]
            XV_mini_batch = x["XV"]  # [B, num_heads, K, head_dim]
            eta_mini_batch = x["eta"]  # [B, num_heads, K, K]
            token_eta_mini_batch = x["token_eta"]  # [B, num_heads, K, K]
            ttt_lr_eta_mini_batch = x["ttt_lr_eta"]  # [B, num_heads, K, K]

            # Forward pass
            X1 = XK_mini_batch  # [B, num_heads, K, head_dim]
            Z1 = torch.matmul(X1, W1_init) + b1_init  # [B, num_heads, K, 4*head_dim]
            X2 = F.gelu(Z1, approximate="tanh")  # [B, num_heads, K, 4*head_dim]
            Z2 = torch.matmul(X2, W2_init) + b2_init  # [B, num_heads, K, head_dim]
            reconstruction_target = XV_mini_batch - XK_mini_batch  # [B, num_heads, K, head_dim]

            ln_weight = self.ttt_norm_weight.reshape(1, self.num_heads, 1, self.head_dim)  # [1, nh, 1, f]
            ln_bias = self.ttt_norm_bias.reshape(1, self.num_heads, 1, self.head_dim)      # [1, nh, 1, f]
    
            # Compute gradients
            grad_l_wrt_Z2 = ln_fused_l2_bwd(Z2, reconstruction_target, ln_weight, ln_bias)  # [B, nh, K, f]
            grad_l_wrt_Z1 = torch.matmul(grad_l_wrt_Z2, W2_init.transpose(-2, -1)) * gelu_bwd(Z1)  # [B, nh, K, 4f]

            if self.use_dual_form:
                # Attention computations
                Attn1 = torch.tril(torch.matmul(XQ_mini_batch, X1.transpose(-2, -1)))  # [B, nh, K, K]
        
                # Compute b1_bar
                b1_bar = b1_init - torch.matmul(Attn1, grad_l_wrt_Z1)  # [B, nh, 1, 4*head_dim]
    
                # Compute Z1_bar
                # eta_mini_batch: [B, nh, K, K]
                # X1: [B, nh, K, head_dim=32]
                # grad_l_wrt_Z1: [B, nh, K, 4*head_dim=128]
                Z1_bar = torch.matmul(XQ_mini_batch,W1_init) - torch.matmul((eta_mini_batch * Attn1),grad_l_wrt_Z1) + b1_bar
                X2_bar = F.gelu(Z1_bar, approximate="tanh")

                Attn2 = torch.tril(torch.matmul(X2_bar,X2.transpose(-2, -1)))
                b2_bar = b2_init - torch.matmul(Attn2, grad_l_wrt_Z2)
                Z2_bar = torch.matmul(X2_bar,W2_init) - torch.matmul((eta_mini_batch * Attn2),grad_l_wrt_Z2) + b2_bar

                last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
                #print("eta_mini_batch",eta_mini_batch.shape)
                #print("grad_l_wrt_Z1",grad_l_wrt_Z1.shape)
                # Compute weighted_X1 = eta_mini_batch @ X1 --> [B, nh, K, 32]
                weighted_X1 = torch.matmul(eta_mini_batch, X1)  # [B, nh, K=64, head_dim=32]
                
                # Compute W1_update = weighted_X1.transpose(2,3) @ grad_l_wrt_Z1 --> [B, nh, 32, 128]
                W1_update = torch.matmul(weighted_X1.transpose(2, 3), grad_l_wrt_Z1)  # [B, nh, 32, 128]
                W1_last = W1_init - W1_update  # [B, nh, 32, 128]
    
                # Compute b1_update = sum(weighted_X1 * grad_l_wrt_Z1, dim=2, keepdim=True) --> [B, nh, 1, 128]
                b1_update = torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)  # [B, nh, 1, 128]
                b1_last = b1_init - b1_update  # [B, nh, 1, 128]
    
                # Compute weighted_X2_bar = eta_mini_batch @ X2_bar --> [B, nh, K=64, 128]
                weighted_X2_bar = torch.matmul(eta_mini_batch, X2_bar)  # [B, nh, 64, 128]

                # Compute W2_update = weighted_X2_bar.transpose(2,3) @ grad_l_wrt_Z2 --> [B, nh, 128, 32]
                W2_update = torch.matmul(weighted_X2_bar.transpose(2, 3), grad_l_wrt_Z2)  # [B, nh, 128, 32]
                W2_last = W2_init - W2_update  # [B, nh, 128, 32]
    
                # Compute b2_update = sum(weighted_X2_bar * grad_l_wrt_Z2, dim=2, keepdim=True) --> [B, nh, 1, 32]
                b2_update = torch.sum(last_eta_mini_batch * grad_l_wrt_Z2, dim=-2, keepdim=True)  # [B, nh, 1, 32]
                b2_last = b2_init - b2_update  # [B, nh, 1, 32]
    
                # Zero gradients for the last parameters
                params_dict["W1_grad"] = torch.zeros_like(W1_last)
                params_dict["b1_grad"] = torch.zeros_like(b1_last)
                params_dict["W2_grad"] = torch.zeros_like(W2_last)
                params_dict["b2_grad"] = torch.zeros_like(b2_last)
    
                # Update the parameter states
                params_dict["W1_states"] = W1_last.detach()
                params_dict["b1_states"] = b1_last.detach()
                params_dict["W2_states"] = W2_last.detach()
                params_dict["b2_states"] = b2_last.detach()
    
                # Return the output for this mini_batch
                Z2_bar = ln_fwd(Z2_bar, ln_weight, ln_bias)
                XQW_mini_batch = XQ_mini_batch + Z2_bar
                #print("XQW_mini_batch",XQW_mini_batch)
                #print("params_dict",params_dict)
                return params_dict,XQW_mini_batch  # Return y
    
            else:
                # 如果不使用双重形式，这里需要实现对应的逻辑
                # 目前未实现，因此抛出异常
                raise NotImplementedError("Primal form is not implemented.")
        XQW_batch = torch.empty(
            (num_mini_batch, B, self.num_heads, mini_batch_size, self.head_dim),
            device=device,
            dtype=dtype,
        )
        # XQW_batch: [num_mini_batch, B, num_heads, mini_batch_size, head_dim]
        batch_params_dict, XQW_batch = scan(
            compute_mini_batch,
            init_params_dict,
            inputs,
            XQW_batch,
            self.config.scan_checkpoint_group_size if self.training else 0,
        )
        #[B, num_heads, L, C]
        if cache_params is not None:
            cache_params.update(batch_params_dict, self.layer_idx, L)

        # [num_mini_batch, B, num_heads, mini_batch_size, head_dim] -> [B, num_mini_batch, mini_batch_size, num_heads, head_dim]
        XQW_batch = XQW_batch.permute(1, 0, 3, 2, 4)
        # [B, L, C]
        XQW_batch = XQW_batch.reshape(B, L, self.width)
        return XQW_batch, batch_params_dict


    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        cache_params=None,
    ):
        # Step 1: Get projections
        XQ, XK, XV = self.get_qkv_projections(hidden_states, cache_params=cache_params)

        # Step 2: Reshape projections for 2D data
        B, N, D = hidden_states.shape
        H_p = W_p = int(N ** 0.5)
        XQ = XQ.view(B, -1, H_p, W_p, self.head_dim)  # [B, num_heads, H_p, W_p, head_dim]
        XK = XK.view(B, -1, H_p, W_p, self.head_dim)
        XV = XV.view(B, -1, H_p, W_p, self.head_dim)

        # Prepare inputs for get_ttt_inputs
        inputs = {
            "XQ": XQ,  # [B, num_heads, H_p, W_p, head_dim]
            "XK": XK,  # [B, num_heads, H_p, W_p, head_dim]
            "XV": XV,  # [B, num_heads, H_p, W_p, head_dim]
            "X": hidden_states,  # [B, N, C]
        }

        # Get ttt_inputs
        ttt_inputs = self.get_ttt_inputs(inputs, self.mini_batch_size, cache_params)

        # Call ttt method
        output, updated_params = self.ttt(
            inputs=ttt_inputs,
            mini_batch_size=self.mini_batch_size,
            last_mini_batch_params_dict=None,  # Adjust if you have caching
            cache_params=cache_params,
        )

        if output is None:
            raise ValueError("TTT method returned None. Please check the ttt() method implementation.")

        # Apply output projection
        output = self.o_proj(output)  # [B, L, hidden_size]

        # **添加调试打印语句**
        #print(f"After o_proj, output shape: {output.shape}")

        # Return output and updated_params
        return output, updated_params
