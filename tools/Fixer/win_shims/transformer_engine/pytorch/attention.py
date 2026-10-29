import torch
import torch.nn.functional as F


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs, tensor_format="sbhd", fused=False, **kw):
    # Same math as TE's non-interleaved RoPE. freqs is [s, 1, 1, d].
    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)  # [1, s, 1, d]
    d = freqs.shape[-1]
    t_rot, t_pass = t[..., :d], t[..., d:]
    cos, sin = freqs.cos(), freqs.sin()
    out = (t_rot.float() * cos + _rotate_half(t_rot.float()) * sin).to(t.dtype)
    return torch.cat((out, t_pass), dim=-1) if t_pass.shape[-1] else out


class DotProductAttention(torch.nn.Module):
    def __init__(self, num_heads, head_dim, qkv_format="bshd", **kw):
        super().__init__()
        assert qkv_format == "bshd", qkv_format

    def forward(self, q, k, v, **kw):  # [B, S, H, D] -> [B, S, H*D]
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return o.transpose(1, 2).flatten(2)
