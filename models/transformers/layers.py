import xformers
import xformers.ops
from einops import rearrange, repeat

import torch
import torch.nn.functional as F
from torch import nn, einsum

from utils.typing import *


class CustomUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv_transpose=True):
        super(CustomUpsample, self).__init__()
        self.use_conv_transpose = use_conv_transpose

        if self.use_conv_transpose:
            self.upsample_layer = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            )
        else:
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        if self.use_conv_transpose:
            x = self.upsample_layer(x)
        else:
            x = self.upsample_layer(x)
            x = self.conv_layer(x)
        return x


class MemoryEfficientCrossAttention(nn.Module):
    """
    Follow Stable Diffusion implementation: https://github.com/Stability-AI/stablediffusion/blob/cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf/ldm/modules/attention.py#L197
    """

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        if context_dim is None:
            context_dim = query_dim
        # print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and "
        #       f"inner_dim: {inner_dim}, num_heads: {heads}")

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        if context is None:
            context = x  # self attention when context is None
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        q, k, v = map(lambda t: t.contiguous(), (q, k, v))
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)
        if mask is not None:
            raise NotImplementedError
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h).contiguous()
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        if context_dim is None:
            context_dim = query_dim

        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            raise ValueError("Flash Attention requires PyTorch >= 2.0")

    def forward(self, x, context=None, mask=None):
        """
        If context is None, apply self_attention instead.
        """
        h = self.heads
        q = self.to_q(x)
        if context is None:
            context = x  # self attention when context is None
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q, k, v = map(lambda t: t.contiguous(), (q, k, v))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h).contiguous()
        return self.to_out(out)


class GEGLU(nn.Module):
    """https://github.com/Stability-AI/stablediffusion/blob/cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf/ldm/modules/attention.py#L49"""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """https://github.com/Stability-AI/stablediffusion/blob/cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf/ldm/modules/attention.py#L59"""
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        if dim_out is None:
            dim_out = dim
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True):
        super().__init__()
        attn_block = MemoryEfficientCrossAttention

        self.self_attn = attn_block(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.cross_attn = attn_block(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head,
                                     dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.cross_attn(self.norm1(x), context=context) + x
        x = self.self_attn(self.norm2(x)) + x
        x = self.ff(self.norm3(x)) + x
        return x


class BasicSelfAttnBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., gated_ff=True):
        super().__init__()
        attn_block = MemoryEfficientCrossAttention

        self.self_attn = attn_block(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.self_attn(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        return x


class PSUpsampler(nn.Module):
    def __init__(
        self,
        d_model,
        d_model_out,
        scale_factor,
    ):
        super().__init__()
        self.proj = nn.Conv2d(d_model, d_model_out * (scale_factor **2), kernel_size=1)
        self.pixelshuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        """
        Args:
            x (Tensor): input, (batch_size, channel, height, width)

        Returns:
            x (Tensor): output, (batch_size, out_channel, height*scale_factor, width*scale_factor)
        """
        assert x.ndim == 4
        x = self.proj(x)
        x = self.pixelshuffle(x)
        return x