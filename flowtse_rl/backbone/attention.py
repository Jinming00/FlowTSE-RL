import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import einops
import os
from einops import rearrange, repeat
from inspect import isfunction
from .rotary import RotaryEmbedding


_requested_attention_mode = os.environ.get('MEANFLOW_ATTENTION_MODE', 'auto').lower()
if _requested_attention_mode not in {'auto', 'flash', 'mem_efficient', 'math'}:
    raise ValueError('MEANFLOW_ATTENTION_MODE must be auto, flash, mem_efficient, or math')
if _requested_attention_mode == 'auto':
    ATTENTION_MODE = 'flash' if hasattr(nn.functional, 'scaled_dot_product_attention') else 'math'
elif _requested_attention_mode == 'flash' and not hasattr(nn.functional, 'scaled_dot_product_attention'):
    raise RuntimeError('Flash attention was requested but scaled_dot_product_attention is unavailable')
else:
    ATTENTION_MODE = _requested_attention_mode
if ATTENTION_MODE == 'mem_efficient':
    if not hasattr(torch.backends, 'cuda') or not hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        raise RuntimeError('Memory-efficient SDPA is unavailable in this PyTorch build')
    # Select the requested memory-efficient SDPA backend explicitly.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
print(f'attention mode is {ATTENTION_MODE}')


def add_mask(sim, mask):
    b, ndim = sim.shape[0], mask.ndim
    if ndim == 3:
        mask = rearrange(mask, "b n m -> b 1 n m")
    if ndim == 2:
        mask = repeat(mask, "n m -> b 1 n m", b=b)
    max_neg_value = -torch.finfo(sim.dtype).max
    sim = sim.masked_fill(~mask, max_neg_value)
    return sim


def create_mask(q_shape, k_shape, device, q_mask=None, k_mask=None):
    def default(val, d):
        return val if val is not None else (d() if isfunction(d) else d)
    b, i, j, device = q_shape[0], q_shape[-2], k_shape[-2], device
    q_mask = default(q_mask, torch.ones((b, i), device=device, dtype=torch.bool))
    k_mask = default(k_mask, torch.ones((b, j), device=device, dtype=torch.bool))
    attn_mask = rearrange(q_mask, 'b i -> b 1 i 1') * rearrange(k_mask, 'b j -> b 1 1 j')
    return attn_mask


class Attention(nn.Module):
    def __init__(self, dim, context_dim=None, num_heads=8,
                 qkv_bias=False, qk_scale=None, qk_norm='layernorm',
                 attn_drop=0., proj_drop=0., rope_mode='shared'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if context_dim is None:
            self.cross_attn = False
        else:
            self.cross_attn = True

        context_dim = dim if context_dim is None else context_dim

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(context_dim, dim, bias=qkv_bias)

        if qk_norm is None:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        elif qk_norm == 'layernorm':
            self.norm_q = nn.LayerNorm(head_dim)
            self.norm_k = nn.LayerNorm(head_dim)
        else:
            raise NotImplementedError

        self.attn_drop_p = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.cross_attn:
            assert rope_mode == 'none'
        self.rope_mode = rope_mode
        if self.rope_mode == 'shared' or self.rope_mode == 'x_only':
            self.rotary = RotaryEmbedding(dim=head_dim)
        elif self.rope_mode == 'dual':
            self.rotary_x = RotaryEmbedding(dim=head_dim)
            self.rotary_c = RotaryEmbedding(dim=head_dim)

    def _rotary(self, q, k, extras):
        if self.rope_mode == 'shared':
            q, k = self.rotary(q=q, k=k)
        elif self.rope_mode == 'x_only':
            q_x, k_x = self.rotary(q=q[:, :, extras:, :], k=k[:, :, extras:, :])
            q_c, k_c = q[:, :, :extras, :], k[:, :, :extras, :]
            q = torch.cat((q_c, q_x), dim=2)
            k = torch.cat((k_c, k_x), dim=2)
        elif self.rope_mode == 'dual':
            q_x, k_x = self.rotary_x(q=q[:, :, extras:, :], k=k[:, :, extras:, :])
            q_c, k_c = self.rotary_c(q=q[:, :, :extras, :], k=k[:, :, :extras, :])
            q = torch.cat((q_c, q_x), dim=2)
            k = torch.cat((k_c, k_x), dim=2)
        elif self.rope_mode == 'none':
            pass
        else:
            raise NotImplementedError
        return q, k

    def _attn(self, q, k, v, mask_binary):
        if ATTENTION_MODE in {'flash', 'mem_efficient'}:
            x = F.scaled_dot_product_attention(q, k, v,
                                               dropout_p=self.attn_drop_p,
                                               attn_mask=mask_binary)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'math':
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = add_mask(attn, mask_binary) if mask_binary is not None else attn
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        else:
            raise NotImplementedError
        return x

    def forward(self, x, context=None, context_mask=None, extras=0, attention_mask=None):
        B, L, C = x.shape
        if context is None:
            context = x

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        if context_mask is not None:
            # A key mask broadcasts across heads and queries without allocating
            # a dense [batch, query, key] mask for long utterances.
            mask_binary = context_mask[:, None, None, :].bool()
        else:
            mask_binary = None

        if attention_mask is not None:
            pairwise = attention_mask[:, None].bool()
            mask_binary = pairwise if mask_binary is None else pairwise & mask_binary

        q = einops.rearrange(q, 'B L (H D) -> B H L D', H=self.num_heads)
        k = einops.rearrange(k, 'B L (H D) -> B H L D', H=self.num_heads)
        v = einops.rearrange(v, 'B L (H D) -> B H L D', H=self.num_heads)

        q = self.norm_q(q)
        k = self.norm_k(k)

        q, k = self._rotary(q, k, extras)

        x = self._attn(q, k, v, mask_binary)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
