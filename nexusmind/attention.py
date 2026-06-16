"""Hybrid attention module for NexusMind.

Implements the Xiaomi MiMo–style interleaved attention pattern that
alternates between efficient *sliding-window* (local) attention and
full *global* (causal) attention across transformer layers.

Module inventory
────────────────
• RotaryEmbedding        — RoPE positional encoding (complex-number impl)
• SlidingWindowAttention  — Causal attention restricted to a local window
• GlobalAttention         — Standard full causal multi-head attention
• HybridAttention         — Layer-index–aware wrapper that dispatches to
                            sliding-window or global depending on layer id

All modules support:
  ✓ Causal masking
  ✓ Multi-head attention (head_dim = d_model // n_heads)
  ✓ KV cache for autoregressive inference
  ✓ Flash-Attention–compatible shapes (B, n_heads, S, head_dim)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nexusmind.config import NexusConfig


# =========================================================================== #
#  Rotary Positional Embedding (RoPE)                                          #
# =========================================================================== #
class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Encodes absolute position information via rotation matrices applied
    to query/key pairs.  This implementation pre-computes the complex
    exponentials for the maximum sequence length and caches them as a
    non-gradient buffer for efficiency.

    The rotation is applied in the complex plane: each consecutive pair
    of dimensions is treated as a 2-D vector and rotated by a position-
    dependent angle, giving the model translation-equivariant relative
    position sensitivity.

    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary
    Position Embedding", 2021.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Inverse frequencies: θ_i = base^{-2i/d} for i in [0, d/2)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build and register the cos/sin cache for positions [0, seq_len)."""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (S, D/2)
        # Duplicate for full head_dim: [θ0, θ1, …, θ0, θ1, …]
        emb = torch.cat([freqs, freqs], dim=-1)  # (S, D)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, q: Tensor, k: Tensor, offset: int = 0
    ) -> Tuple[Tensor, Tensor]:
        """Apply rotary embeddings to queries and keys.

        Args:
            q: ``(B, H, S, D)`` query tensor.
            k: ``(B, H, S, D)`` key tensor.
            offset: Position offset for KV-cache continuation.

        Returns:
            Tuple of rotated ``(q, k)`` with the same shapes.
        """
        S = q.size(2)
        # Extend cache if needed
        if offset + S > self.cos_cached.size(0):
            self._build_cache(offset + S)

        cos = self.cos_cached[offset : offset + S].unsqueeze(0).unsqueeze(0)  # (1,1,S,D)
        sin = self.sin_cached[offset : offset + S].unsqueeze(0).unsqueeze(0)

        q_rot = (q * cos) + (_rotate_half(q) * sin)
        k_rot = (k * cos) + (_rotate_half(k) * sin)
        return q_rot, k_rot


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate pairs of dimensions: [x0, x1, x2, x3, …] → [-x1, x0, -x3, x2, …]."""
    x1 = x[..., : x.size(-1) // 2]
    x2 = x[..., x.size(-1) // 2 :]
    return torch.cat([-x2, x1], dim=-1)


# =========================================================================== #
#  Causal Mask Utilities                                                       #
# =========================================================================== #
def _make_causal_mask(
    q_len: int,
    kv_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Full causal mask: position i can attend to positions [0, i].

    Returns:
        ``(1, 1, q_len, kv_len)`` additive mask with 0 for allowed
        positions and ``-inf`` for masked positions.
    """
    mask = torch.full((q_len, kv_len), float("-inf"), device=device, dtype=dtype)
    # offset = kv_len - q_len handles the KV-cache case where kv_len > q_len
    offset = kv_len - q_len
    mask = torch.triu(mask, diagonal=1 + offset)
    return mask.unsqueeze(0).unsqueeze(0)


def _make_sliding_window_mask(
    q_len: int,
    kv_len: int,
    window_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Sliding-window causal mask.

    Position *i* can attend to positions ``[max(0, i - window + 1), i]``
    (inclusive), where positions are measured in the full KV sequence.

    Returns:
        ``(1, 1, q_len, kv_len)`` additive mask.
    """
    # Row indices in query space; column indices in full kv space
    offset = kv_len - q_len
    rows = torch.arange(q_len, device=device).unsqueeze(1) + offset  # (Q, 1)
    cols = torch.arange(kv_len, device=device).unsqueeze(0)          # (1, KV)

    # Causal: col <= row
    causal = cols <= rows
    # Window: col >= row - window_size + 1
    window = cols >= (rows - window_size + 1)

    valid = causal & window
    mask = torch.where(
        valid,
        torch.tensor(0.0, device=device, dtype=dtype),
        torch.tensor(float("-inf"), device=device, dtype=dtype),
    )
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, Q, KV)


# =========================================================================== #
#  KV Cache                                                                    #
# =========================================================================== #
class KVCache:
    """Simple key/value cache for autoregressive generation.

    Stores accumulated K and V tensors of shape ``(B, H, S_cached, D)``
    and provides an ``update`` method that appends new K/V slices.
    """

    def __init__(self) -> None:
        self.k: Optional[Tensor] = None
        self.v: Optional[Tensor] = None

    @property
    def seq_len(self) -> int:
        """Number of cached positions."""
        return 0 if self.k is None else self.k.size(2)

    def update(self, new_k: Tensor, new_v: Tensor) -> Tuple[Tensor, Tensor]:
        """Append new keys/values and return full cached tensors.

        Args:
            new_k: ``(B, H, S_new, D)``
            new_v: ``(B, H, S_new, D)``

        Returns:
            Tuple of full ``(k, v)`` each ``(B, H, S_total, D)``.
        """
        if self.k is None:
            self.k = new_k
            self.v = new_v
        else:
            self.k = torch.cat([self.k, new_k], dim=2)
            self.v = torch.cat([self.v, new_v], dim=2)
        return self.k, self.v

    def reset(self) -> None:
        """Clear the cache."""
        self.k = None
        self.v = None


# =========================================================================== #
#  Sliding-Window Attention                                                    #
# =========================================================================== #
class SlidingWindowAttention(nn.Module):
    """Local causal attention restricted to a sliding window.

    Each query position attends only to the most recent
    ``window_size`` key positions (including itself).  This bounds
    memory and compute to O(S × W) instead of O(S²) while preserving
    strong short-range modelling.

    Compatible with KV cache for autoregressive inference.  During
    cached decoding only the visible window of the KV cache is used.
    """

    def __init__(self, config: NexusConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.window_size = config.sliding_window_size
        self.dropout = config.dropout

        assert config.d_model % config.n_heads == 0, (
            f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})"
        )

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len)
        self.attn_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[Tensor, Optional[KVCache]]:
        """Sliding-window self-attention.

        Args:
            x: ``(B, S, D)`` input tensor.
            kv_cache: Optional KV cache for autoregressive decoding.

        Returns:
            output: ``(B, S, D)``
            kv_cache: Updated cache (same object if provided, else new).
        """
        B, S, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, S, H, Dh).transpose(1, 2)  # (B, H, S, Dh)
        k = self.k_proj(x).view(B, S, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, Dh).transpose(1, 2)

        # Cache handling
        if kv_cache is None:
            kv_cache = KVCache()
        offset = kv_cache.seq_len

        # Apply RoPE before caching
        q, k = self.rope(q, k, offset=offset)

        # Update cache
        k, v = kv_cache.update(k, v)
        kv_len = k.size(2)

        # Sliding-window causal mask
        mask = _make_sliding_window_mask(
            S, kv_len, self.window_size, device=x.device, dtype=x.dtype
        )

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(Dh)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,H,S,KV)
        attn_weights = attn_weights + mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # (B, H, S, Dh)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.o_proj(out)

        return out, kv_cache


# =========================================================================== #
#  Global (Full Causal) Attention                                              #
# =========================================================================== #
class GlobalAttention(nn.Module):
    """Standard full causal multi-head attention.

    Every query position can attend to all preceding key positions
    (including itself), providing unbounded context at the cost of
    O(S²) memory.  Used at periodic intervals in the hybrid layout
    to propagate long-range information.
    """

    def __init__(self, config: NexusConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout

        assert config.d_model % config.n_heads == 0

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len)
        self.attn_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[Tensor, Optional[KVCache]]:
        """Full causal self-attention.

        Args:
            x: ``(B, S, D)``
            kv_cache: Optional KV cache.

        Returns:
            output ``(B, S, D)`` and updated cache.
        """
        B, S, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, S, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, S, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, Dh).transpose(1, 2)

        if kv_cache is None:
            kv_cache = KVCache()
        offset = kv_cache.seq_len

        q, k = self.rope(q, k, offset=offset)
        k, v = kv_cache.update(k, v)
        kv_len = k.size(2)

        # Full causal mask
        mask = _make_causal_mask(S, kv_len, device=x.device, dtype=x.dtype)

        scale = 1.0 / math.sqrt(Dh)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = attn_weights + mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.o_proj(out)

        return out, kv_cache


# =========================================================================== #
#  Hybrid Attention Dispatcher                                                 #
# =========================================================================== #
class HybridAttention(nn.Module):
    """Interleaves sliding-window and global attention across layers.

    Pattern (MiMo):
        Every ``N``-th layer uses full global attention; all other layers
        use sliding-window attention.  ``N`` is derived from
        ``config.global_attention_ratio``:

            global_every = round(1 / global_attention_ratio)

        Example with 12 layers and ratio 0.25 → global at layers 3, 7, 11.

    This balances long-range information flow (global layers) with
    computational efficiency (window layers), giving near-linear
    scaling in sequence length while retaining strong long-context
    performance.

    Args:
        config: NexusMind configuration.
        layer_idx: Zero-based index of this layer in the transformer stack.
    """

    def __init__(self, config: NexusConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        # Determine if this layer is global or local
        global_every = max(1, round(1.0 / config.global_attention_ratio))
        self.is_global = ((layer_idx + 1) % global_every == 0)

        if self.is_global:
            self.attn = GlobalAttention(config)
        else:
            self.attn = SlidingWindowAttention(config)

    def forward(
        self,
        x: Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[Tensor, Optional[KVCache]]:
        """Dispatch to the correct attention type.

        Args:
            x: ``(B, S, D)``
            kv_cache: Optional KV cache.

        Returns:
            output ``(B, S, D)`` and updated cache.
        """
        return self.attn(x, kv_cache=kv_cache)

    def extra_repr(self) -> str:
        kind = "global" if self.is_global else f"sliding_window"
        return f"layer_idx={self.layer_idx}, type={kind}"
