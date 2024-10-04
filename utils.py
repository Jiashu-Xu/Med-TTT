# utils.py

import torch

def ln_fwd(x, gamma, beta, eps=1e-6):
    """
    LayerNorm forward pass.
    Args:
        x: [B, nh, K, f]
        gamma: [1, nh, 1, f]
        beta: [1, nh, 1, f]
    Returns:
        y: [B, nh, K, f]
    """
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std
    y = gamma * x_hat + beta
    return y

def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-6):
    """
    LayerNorm backward pass fused with L2 loss.
    Args:
        x: [B, nh, K, f]
        l2_target: [B, nh, K, f]
        gamma: [1, nh, 1, f]
        beta: [1, nh, 1, f]
    Returns:
        grad_x: [B, nh, K, f]
    """
    D = x.shape[-1]
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std
    y = gamma * x_hat + beta
    grad_output = y - l2_target  # Assuming L2 loss derivative
    grad_x_hat = grad_output * gamma
    z = (
        (1.0 / D)
        * (
            D * grad_x_hat
            - grad_x_hat.sum(dim=-1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
        )
        / std
    )
    return z

def gelu_bwd(x):
    """
    GELU backward pass approximation.
    Args:
        x: [B, nh, K, 4f]
    Returns:
        grad: [B, nh, K, 4f]
    """
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    grad = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return grad

def tree_map(fn, tree):
    """
    Recursively applies a function to all leaves in a tree structure.
    Args:
        fn: function to apply
        tree: nested structure (dict, list, etc.)
    Returns:
        tree with function applied to all leaves
    """
    if isinstance(tree, dict):
        return {k: tree_map(fn, v) for k, v in tree.items()}
    elif isinstance(tree, list):
        return [tree_map(fn, v) for v in tree]
    else:
        return fn(tree)

# utils.py

def scan(f, init, xs, out, checkpoint_group=0):
    """
    Mimics jax.lax.scan function for sequentially applying a function over inputs.
    Args:
        f: function to apply, takes (carry, x) and returns (new_carry, y)
        init: initial carry
        xs: list of inputs (dicts)
        out: list to store outputs
        checkpoint_group: int, number of groups for checkpointing
    Returns:
        carry: final carry
        out: list of outputs
    """
    carry = init
    if isinstance(xs, dict):
        num_items = next(iter(xs.values())).size(2)
    else:
        num_items = len(xs[0])
    def scan_fn(carry, i_start, i_end):
        for i in range(i_start, i_end):
            if isinstance(xs, dict):
                x = {key: tensor[:,:,i,...] for key, tensor in xs.items()}
            else:
                x = [x[:,:,i,...] for x in xs]
            carry, y = f(carry, x)
            out[i] = y
        return carry

    if checkpoint_group > 0:
        ckpt_every_n = max(1, num_items // checkpoint_group)
        for k in range(0, num_items, ckpt_every_n):
            carry = torch.utils.checkpoint.checkpoint(
                scan_fn, carry, k, min(k + ckpt_every_n, num_items), use_reentrant=False
            )
    else:
        carry = scan_fn(carry, 0, num_items)

    return carry, out
