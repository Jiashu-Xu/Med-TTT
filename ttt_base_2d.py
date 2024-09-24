# ttt_base_2d.py

import torch
import torch.nn as nn

class TTTBase2D(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.width = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.width // self.num_heads
        self.mini_batch_size = config.mini_batch_size

        token_idx = 1.0 / torch.arange(1, self.mini_batch_size + 1, dtype=torch.float32)
        self.register_buffer("token_idx", token_idx, persistent=False)
        self.learnable_token_idx = nn.Parameter(torch.zeros((self.mini_batch_size,)))

        self.share_qk = config.share_qk
        self.conv_kernel = config.conv_kernel
        self._init_qkvo_proj()
        self._init_ttt_lr_gate()
        self._init_ttt_ln()

        self.use_gate = config.use_gate
        if self.use_gate:
            self.g_proj = nn.Linear(self.width, self.width, bias=False)

        self.post_norm = nn.LayerNorm(self.width, eps=1e-6)

    def _init_qkvo_proj(self):
        self.q_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        if not self.share_qk:
            self.k_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.width, bias=False)

        if self.share_qk:
            # 修正卷积层的输入输出通道数和 groups 参数
            self.conv_q = nn.Conv2d(
                self.head_dim,  # 输入通道数为 head_dim = 32
                self.head_dim,  # 输出通道数为 head_dim = 32
                kernel_size=self.conv_kernel,
                groups=self.head_dim,  # groups 设置为 head_dim，实现逐通道卷积
                padding=self.conv_kernel // 2,
                bias=True
            )
            self.conv_k = nn.Conv2d(
                self.head_dim,
                self.head_dim,
                kernel_size=self.conv_kernel,
                groups=self.head_dim,
                padding=self.conv_kernel // 2,
                bias=True
            )

    def _init_ttt_lr_gate(self):
        self.ttt_norm_weight = nn.Parameter(torch.ones(self.num_heads, self.head_dim))
        self.ttt_norm_bias = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

    def _init_ttt_ln(self):
        pass  # 已在 _init_ttt_lr_gate 中初始化

    def get_qkv_projections(self, hidden_states, cache_params=None):
        B, N, D = hidden_states.shape
        if self.share_qk:
            xq = self.q_proj(hidden_states)  # [B, N, num_heads * head_dim]
            xv = self.v_proj(hidden_states)  # [B, N, num_heads * head_dim]
            H = W = int(N ** 0.5)
            xq = xq.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
            XQ = self.conv_q(xq.reshape(B * self.num_heads, self.head_dim, H, W))  # [B*num_heads, head_dim, H, W]
            XQ = XQ.view(B, self.num_heads, self.head_dim, -1).transpose(2, 3)  # [B, num_heads, N, head_dim]
            XK = self.conv_k(xq.reshape(B * self.num_heads, self.head_dim, H, W))  # [B*num_heads, head_dim, H, W]
            XK = XK.view(B, self.num_heads, self.head_dim, -1).transpose(2, 3)  # [B, num_heads, N, head_dim]
            XV = xv.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        else:
            XQ = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
            XK = self.k_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
            XV = self.v_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        return XQ, XK, XV

    def get_eta(self, X, mini_batch_step_offset, mini_batch_size):
        """
        Compute eta for attention computations.
        Args:
            X: [B, num_mini_batch, mini_batch_size, C]
            mini_batch_step_offset: int
            mini_batch_size: int
        Returns:
            token_eta: [B, num_heads, K, K]
            ttt_lr_eta: [B, num_heads, K, K]
        """
        B, num_mini_batch, K, C = X.shape

        # Create token_eta as a lower triangular matrix based on token_idx
        # token_idx: [1, 1, K, 1] -> expand to [B, num_heads, K, K]
        token_eta = torch.tril(
            self.token_idx[:K].unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(B, self.num_heads, K, K)
        )  # [B, num_heads, K, K]

        # Create ttt_lr_eta as a fixed learning rate per head
        # Here, we use a scalar learning rate from config.ttt_base_lr
        ttt_lr_eta = torch.full(
            (B, self.num_heads, K, K),
            self.config.ttt_base_lr,
            device=X.device,
            dtype=X.dtype
        )  # [B, num_heads, K, K]

        return token_eta, ttt_lr_eta

    def get_ttt_inputs(self, inputs, mini_batch_size, cache_params):
        """
        Prepare inputs for the TTT method with minibatch processing.
        Args:
            inputs: dict containing "XQ", "XK", "XV", "X"
            mini_batch_size: int
            cache_params: optional cache parameters
        Returns:
            dict containing "XQ", "XK", "XV", "eta", "token_eta", "ttt_lr_eta"
        """
        XQ = inputs["XQ"]  # [B, num_heads, num_mini_batch, K, head_dim]
        XK = inputs["XK"]
        XV = inputs["XV"]
        X = inputs["X"]  # [B, N_patches, C]
        B, L, C = X.shape
        num_mini_batch = L // mini_batch_size
        remainder = L % mini_batch_size
        if remainder != 0:
            num_mini_batch += 1

        # Handle padding if necessary (optional)
        '''''
        if remainder != 0:
            padding_size = mini_batch_size - remainder
            pad = torch.zeros(B, num_mini_batch * mini_batch_size - L, C, device=X.device, dtype=X.dtype)
            X = torch.cat([X, pad], dim=1)
            XQ = torch.cat([XQ, torch.zeros(B, self.num_heads, 1, mini_batch_size, self.head_dim, device=X.device, dtype=X.dtype)], dim=2)
            XK = torch.cat([XK, torch.zeros(B, self.num_heads, 1, mini_batch_size, self.head_dim, device=X.device, dtype=X.dtype)], dim=2)
            XV = torch.cat([XV, torch.zeros(B, self.num_heads, 1, mini_batch_size, self.head_dim, device=X.device, dtype=X.dtype)], dim=2)
        '''''
        # Reshape inputs for minibatches
        X = X.reshape(B, num_mini_batch, mini_batch_size, self.width)

        XQ = XQ.reshape(B, self.num_heads, num_mini_batch, mini_batch_size, self.head_dim)
       #print("XQ",XQ.shape)
        XK = XK.reshape(B, self.num_heads, num_mini_batch, mini_batch_size, self.head_dim)
        XV = XV.reshape(B, self.num_heads, num_mini_batch, mini_batch_size, self.head_dim)

        if cache_params is not None:
            mini_batch_step_offset = cache_params.seqlen_offset % self.mini_batch_size
        else:
            mini_batch_step_offset = 0

        # Compute eta values
        token_eta, ttt_lr_eta = self.get_eta(X, mini_batch_step_offset, mini_batch_size)  # [B, num_heads, K, K]
        
        # Expand eta to include num_mini_batch dimension
        eta = (token_eta * ttt_lr_eta).unsqueeze(2).repeat(1, 1, num_mini_batch, 1, 1)  # [B, num_heads, num_mini_batch, K, K]

        # Prepare ttt_inputs
        ttt_inputs = {
            "XQ": XQ,                    # [B, num_heads, num_mini_batch, K, head_dim]
            "XK": XK,                    # [B, num_heads, num_mini_batch, K, head_dim]
            "XV": XV,                    # [B, num_heads, num_mini_batch, K, head_dim]
            "eta": eta,                  # [B, num_heads, num_mini_batch, K, K]
            "token_eta": token_eta,      # [B, num_heads, K, K]
            "ttt_lr_eta": ttt_lr_eta,    # [B, num_heads, K, K]
        }

        return ttt_inputs

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

        # Return output and updated_params
        return output, updated_params
