"""NexusMind Memory Systems.

Four complementary memory architectures that give the model persistent,
hierarchical, and differentiable storage across reasoning steps:

1. WorkingMemoryBank   – learned key-value scratchpad (Supermix v19)
2. EpisodicMemory      – NTM-style differentiable memory matrix (Supermix v22)
3. LatentKnowledgeCore – implicit RAG via sparse key-value retrieval (Supermix v20)
4. HierarchicalAbstractionPyramid – 3-level abstraction pyramid (Supermix v21)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity_matrix(
    queries: torch.Tensor,
    keys: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Batched cosine similarity: (B, Q, D) x (B, K, D) -> (B, Q, K)."""
    queries_norm = queries / (queries.norm(dim=-1, keepdim=True) + eps)
    keys_norm = keys / (keys.norm(dim=-1, keepdim=True) + eps)
    return torch.bmm(queries_norm, keys_norm.transpose(1, 2))


# ═══════════════════════════════════════════════════════════════════════════
# 1. Working Memory Bank  (Supermix v19)
# ═══════════════════════════════════════════════════════════════════════════

class WorkingMemoryBank(nn.Module):
    """Learned key-value scratchpad with attention-based read / gated write.

    The memory persists across reasoning steps within a single forward pass
    and is reset between sequences via :meth:`reset`.

    Args:
        d_model: Hidden dimension of the model.
        n_slots: Number of memory slots (key-value pairs).
        n_heads: Number of attention heads for reading.
    """

    def __init__(
        self,
        d_model: int,
        n_slots: int = 32,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_slots = n_slots
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // n_heads

        # Learned initial keys & values
        self.init_keys = nn.Parameter(torch.randn(1, n_slots, d_model) * 0.02)
        self.init_values = nn.Parameter(torch.randn(1, n_slots, d_model) * 0.02)

        # Read projections (multi-head attention over memory)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Write gate: decides how much of new signal to incorporate
        self.write_gate_proj = nn.Linear(d_model * 2, d_model)
        self.write_value_proj = nn.Linear(d_model, d_model)
        self.write_key_proj = nn.Linear(d_model, d_model)

        # Running state (set by reset / write)
        self.register_buffer("_keys", None, persistent=False)
        self.register_buffer("_values", None, persistent=False)

    # ---- lifecycle --------------------------------------------------------

    def reset(self, batch_size: int = 1) -> None:
        """Reset memory to learned initial state for a new sequence."""
        device = self.init_keys.device
        self._keys = self.init_keys.expand(batch_size, -1, -1).clone()  # (B, S, D)
        self._values = self.init_values.expand(batch_size, -1, -1).clone()

    def _ensure_init(self, batch_size: int) -> None:
        if self._keys is None or self._keys.size(0) != batch_size:
            self.reset(batch_size)

    # ---- read -------------------------------------------------------------

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Attention-based retrieval from memory.

        Args:
            query: (B, L, D) or (B, D) query tensor.

        Returns:
            Retrieved information with the same rank as ``query``.
        """
        squeeze = False
        if query.dim() == 2:
            query = query.unsqueeze(1)
            squeeze = True

        B, L, D = query.shape
        self._ensure_init(B)

        H, Dh = self.n_heads, self.head_dim

        q = self.q_proj(query).view(B, L, H, Dh).transpose(1, 2)      # (B, H, L, Dh)
        k = self.k_proj(self._keys).view(B, self.n_slots, H, Dh).transpose(1, 2)
        v = self.v_proj(self._values).view(B, self.n_slots, H, Dh).transpose(1, 2)

        scale = math.sqrt(Dh)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale            # (B, H, L, S)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)                                     # (B, H, L, Dh)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        return out.squeeze(1) if squeeze else out

    # ---- write ------------------------------------------------------------

    def write(self, signal: torch.Tensor) -> None:
        """Gated update of memory values from an external signal.

        The signal is reduced to per-slot updates via content-based
        addressing, then blended with existing values through a learned gate.

        Args:
            signal: (B, L, D) or (B, D) signal to write.
        """
        if signal.dim() == 2:
            signal = signal.unsqueeze(1)

        B = signal.size(0)
        self._ensure_init(B)

        # Content-based addressing: which slots does the signal address?
        addr = _cosine_similarity_matrix(signal, self._keys)             # (B, L, S)
        addr = F.softmax(addr / math.sqrt(self.d_model), dim=-1)

        # Aggregate signal per slot
        slot_update = torch.bmm(addr.transpose(1, 2), signal)           # (B, S, D)

        # Compute gate ∈ [0, 1]
        gate_input = torch.cat([self._values, slot_update], dim=-1)     # (B, S, 2D)
        gate = torch.sigmoid(self.write_gate_proj(gate_input))          # (B, S, D)

        # Gated blend
        new_values = self.write_value_proj(slot_update)
        self._values = gate * self._values + (1 - gate) * new_values

        # Optionally also update keys (slow drift)
        key_delta = torch.tanh(self.write_key_proj(slot_update))
        self._keys = self._keys + 0.01 * key_delta  # small step

    # ---- forward ----------------------------------------------------------

    def forward(
        self,
        query: torch.Tensor,
        write_signal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Read from memory, optionally write first.

        Args:
            query: (B, L, D) query tensor.
            write_signal: Optional (B, L', D) tensor to write before reading.

        Returns:
            (B, L, D) retrieved memory content.
        """
        if write_signal is not None:
            self.write(write_signal)
        return self.read(query)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Episodic Memory  (Supermix v22 – NTM-style)
# ═══════════════════════════════════════════════════════════════════════════

class EpisodicMemory(nn.Module):
    """Differentiable external memory with content-based read/write heads.

    Inspired by Neural Turing Machines, this module maintains a memory matrix
    ``M ∈ R^{n_slots × d_model}`` and provides read/write heads that use
    cosine-similarity based content addressing.

    Args:
        d_model: Hidden dimension.
        n_slots: Number of memory rows.
        n_read_heads: Number of independent read heads.
    """

    def __init__(
        self,
        d_model: int,
        n_slots: int = 64,
        n_read_heads: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_slots = n_slots
        self.n_read_heads = n_read_heads

        # Learned initial memory content
        self.init_memory = nn.Parameter(torch.randn(1, n_slots, d_model) * 0.01)

        # Read head projections (per head)
        self.read_key_projs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_read_heads)
        ])
        self.read_strength_projs = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(n_read_heads)
        ])
        self.read_out = nn.Linear(d_model * n_read_heads, d_model)

        # Write head projections
        self.write_key_proj = nn.Linear(d_model, d_model)
        self.write_strength_proj = nn.Linear(d_model, 1)
        self.erase_proj = nn.Linear(d_model, d_model)
        self.add_proj = nn.Linear(d_model, d_model)

        # Running memory
        self.register_buffer("_memory", None, persistent=False)

    def reset(self, batch_size: int = 1) -> None:
        """Reset to learned initial memory."""
        self._memory = self.init_memory.expand(batch_size, -1, -1).clone()

    def _ensure_init(self, batch_size: int) -> None:
        if self._memory is None or self._memory.size(0) != batch_size:
            self.reset(batch_size)

    def _content_address(
        self,
        key: torch.Tensor,
        strength: torch.Tensor,
    ) -> torch.Tensor:
        """Content-based addressing via cosine similarity.

        Args:
            key: (B, D) lookup key.
            strength: (B, 1) sharpening factor (β).

        Returns:
            (B, S) normalised address weights.
        """
        sim = _cosine_similarity_matrix(
            key.unsqueeze(1), self._memory,
        ).squeeze(1)                                                    # (B, S)
        beta = F.softplus(strength)                                     # ensure β > 0
        return F.softmax(beta * sim, dim=-1)

    # ---- read head --------------------------------------------------------

    def read_head(self, state: torch.Tensor) -> torch.Tensor:
        """Read from memory using content-based addressing.

        Args:
            state: (B, D) controller state.

        Returns:
            (B, D) read vector (concatenation of all heads projected back).
        """
        B = state.size(0)
        self._ensure_init(B)

        reads = []
        for key_proj, str_proj in zip(self.read_key_projs, self.read_strength_projs):
            key = key_proj(state)                                       # (B, D)
            strength = str_proj(state)                                  # (B, 1)
            weights = self._content_address(key, strength)              # (B, S)
            read_vec = torch.bmm(weights.unsqueeze(1), self._memory)   # (B, 1, D)
            reads.append(read_vec.squeeze(1))

        concat = torch.cat(reads, dim=-1)                               # (B, D*H)
        return self.read_out(concat)

    # ---- write head -------------------------------------------------------

    def write_head(
        self,
        state: torch.Tensor,
        erase_signal: Optional[torch.Tensor] = None,
        add_signal: Optional[torch.Tensor] = None,
    ) -> None:
        """Write to memory via content-based erase-then-add.

        Args:
            state: (B, D) controller state used to compute addressing.
            erase_signal: Optional (B, D) explicit erase vector.  If *None*,
                derived from ``state`` via learned projection.
            add_signal: Optional (B, D) explicit add vector.  If *None*,
                derived from ``state`` via learned projection.
        """
        B = state.size(0)
        self._ensure_init(B)

        key = self.write_key_proj(state)
        strength = self.write_strength_proj(state)
        weights = self._content_address(key, strength)                  # (B, S)

        erase = torch.sigmoid(self.erase_proj(state) if erase_signal is None else erase_signal)
        add = torch.tanh(self.add_proj(state) if add_signal is None else add_signal)

        # Erase:  M_t = M_{t-1} * (1 - w_t ⊗ e_t)
        erase_matrix = 1.0 - torch.bmm(
            weights.unsqueeze(2), erase.unsqueeze(1),
        )                                                               # (B, S, D)
        self._memory = self._memory * erase_matrix

        # Add:    M_t = M_t + w_t ⊗ a_t
        add_matrix = torch.bmm(weights.unsqueeze(2), add.unsqueeze(1))
        self._memory = self._memory + add_matrix

    # ---- forward ----------------------------------------------------------

    def forward(
        self,
        state: torch.Tensor,
        do_write: bool = True,
    ) -> torch.Tensor:
        """Read from memory; optionally write the state first.

        Args:
            state: (B, D) controller state.
            do_write: Whether to write before reading.

        Returns:
            (B, D) read vector.
        """
        if do_write:
            self.write_head(state)
        return self.read_head(state)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Latent Knowledge Core  (Supermix v20 – Implicit RAG)
# ═══════════════════════════════════════════════════════════════════════════

class LatentKnowledgeCore(nn.Module):
    """Implicit RAG via a large sparse key-value memory.

    Before reasoning begins the model queries this memory with the input
    representation and retrieves factual / world-knowledge context.  The
    retrieval uses top-k sparse attention to scale to very large memory
    banks without full quadratic cost.

    Args:
        d_model: Hidden dimension.
        n_entries: Number of knowledge entries.
        top_k: Number of entries to retrieve per query token.
        n_heads: Multi-head attention heads for retrieval.
    """

    def __init__(
        self,
        d_model: int,
        n_entries: int = 1024,
        top_k: int = 32,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_entries = n_entries
        self.top_k = min(top_k, n_entries)
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads

        # Knowledge bank (learned parameters – "implicit" RAG)
        self.knowledge_keys = nn.Parameter(
            torch.randn(n_entries, d_model) * 0.02
        )
        self.knowledge_values = nn.Parameter(
            torch.randn(n_entries, d_model) * 0.02
        )

        # Query projection
        self.q_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Gating: blend retrieved knowledge with original input
        self.gate_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retrieve knowledge and blend with input.

        Args:
            x: (B, L, D) input features.

        Returns:
            (B, L, D) knowledge-augmented features.
        """
        B, L, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        queries = self.q_proj(x)                                         # (B, L, D)

        # Compute scores against all keys (efficient for moderate n_entries)
        # queries: (B, L, D),  keys: (N, D)
        scores = torch.einsum("bld,nd->bln", queries, self.knowledge_keys)  # (B, L, N)
        scores = scores / math.sqrt(D)

        # Top-k sparsification
        topk_scores, topk_idx = scores.topk(self.top_k, dim=-1)         # (B, L, K)
        topk_weights = F.softmax(topk_scores, dim=-1)                    # (B, L, K)

        # Gather corresponding values
        # Expand indices for gathering: (B, L, K) -> (B, L, K, D)
        idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        kv = self.knowledge_values.unsqueeze(0).unsqueeze(0).expand(B, L, -1, D)
        topk_values = torch.gather(kv, 2, idx_expanded)                  # (B, L, K, D)

        # Weighted sum of retrieved values
        retrieved = torch.einsum("blk,blkd->bld", topk_weights, topk_values)

        retrieved = self.out_proj(retrieved)

        # Gated fusion
        gate = torch.sigmoid(self.gate_proj(torch.cat([x, retrieved], dim=-1)))
        return x + gate * retrieved

    def reset(self, batch_size: int = 1) -> None:
        """No-op for API compatibility; knowledge bank is stateless."""
        pass


# ═══════════════════════════════════════════════════════════════════════════
# 4. Hierarchical Abstraction Pyramid  (Supermix v21)
# ═══════════════════════════════════════════════════════════════════════════

class _AbstractionLevel(nn.Module):
    """A single level in the abstraction pyramid with its own knowledge bank."""

    def __init__(self, d_model: int, n_slots: int = 16) -> None:
        super().__init__()
        self.bank_keys = nn.Parameter(torch.randn(1, n_slots, d_model) * 0.02)
        self.bank_values = nn.Parameter(torch.randn(1, n_slots, d_model) * 0.02)
        self.query_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend to level-specific knowledge bank.

        Args:
            x: (B, L, D)

        Returns:
            (B, L, D) refined representation.
        """
        B, L, D = x.shape
        q = self.query_proj(x)                                            # (B, L, D)
        keys = self.bank_keys.expand(B, -1, -1)
        vals = self.bank_values.expand(B, -1, -1)

        attn = torch.bmm(q, keys.transpose(1, 2)) / math.sqrt(D)
        attn = F.softmax(attn, dim=-1)
        ctx = torch.bmm(attn, vals)
        return self.norm(x + self.out_proj(ctx))


class HierarchicalAbstractionPyramid(nn.Module):
    """Three-level abstraction pyramid: concrete → abstract → meta.

    Information flows **bottom-up** (projection) and **top-down**
    (distillation bridges) so that high-level reasoning can inform
    concrete representations and vice-versa.

    Args:
        d_model: Hidden dimension.
        n_slots_per_level: Knowledge bank slots per level.
    """

    LEVELS = ("concrete", "abstract", "meta")

    def __init__(
        self,
        d_model: int,
        n_slots_per_level: int = 16,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Per-level knowledge banks
        self.levels = nn.ModuleDict({
            name: _AbstractionLevel(d_model, n_slots_per_level)
            for name in self.LEVELS
        })

        # Bottom-up projections (concrete → abstract, abstract → meta)
        self.up_proj_0 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.up_proj_1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Top-down distillation bridges (meta → abstract, abstract → concrete)
        self.down_bridge_0 = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.down_bridge_1 = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through the three-level pyramid.

        Args:
            x: (B, L, D) input features (treated as concrete level input).

        Returns:
            (B, L, D) fused multi-level representation.
        """
        # --- bottom-up pass ---
        concrete = self.levels["concrete"](x)                            # (B, L, D)
        abstract_in = self.up_proj_0(concrete)
        abstract = self.levels["abstract"](abstract_in)                  # (B, L, D)
        meta_in = self.up_proj_1(abstract)
        meta = self.levels["meta"](meta_in)                              # (B, L, D)

        # --- top-down distillation ---
        abstract_refined = self.down_bridge_0(
            torch.cat([abstract, meta], dim=-1),
        )                                                                # (B, L, D)
        concrete_refined = self.down_bridge_1(
            torch.cat([concrete, abstract_refined], dim=-1),
        )                                                                # (B, L, D)

        # --- fusion ---
        fused = self.fusion(
            torch.cat([concrete_refined, abstract_refined, meta], dim=-1),
        )
        return fused

    def reset(self, batch_size: int = 1) -> None:
        """No-op for API consistency; the pyramid is stateless per-call."""
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: unified memory interface
# ═══════════════════════════════════════════════════════════════════════════

class UnifiedMemory(nn.Module):
    """Convenience wrapper that bundles all four memory systems.

    Args:
        d_model: Hidden dimension.
        wm_slots: Working memory slots.
        em_slots: Episodic memory slots.
        lk_entries: Latent knowledge entries.
        hap_slots: Abstraction pyramid slots per level.
    """

    def __init__(
        self,
        d_model: int,
        wm_slots: int = 32,
        em_slots: int = 64,
        lk_entries: int = 1024,
        hap_slots: int = 16,
    ) -> None:
        super().__init__()
        self.working = WorkingMemoryBank(d_model, n_slots=wm_slots)
        self.episodic = EpisodicMemory(d_model, n_slots=em_slots)
        self.knowledge = LatentKnowledgeCore(d_model, n_entries=lk_entries)
        self.pyramid = HierarchicalAbstractionPyramid(d_model, n_slots_per_level=hap_slots)

    def reset(self, batch_size: int = 1) -> None:
        """Reset all stateful memories."""
        self.working.reset(batch_size)
        self.episodic.reset(batch_size)
        self.knowledge.reset(batch_size)
        self.pyramid.reset(batch_size)

    def forward(
        self,
        x: torch.Tensor,
        write_signal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run full memory pipeline.

        1. Retrieve latent knowledge  (pre-reasoning augmentation)
        2. Pass through abstraction pyramid
        3. Read from working memory (optionally writing first)
        4. Read from episodic memory (and write)

        Args:
            x: (B, L, D) input.
            write_signal: Optional signal to write into working memory.

        Returns:
            (B, L, D) memory-augmented representation.
        """
        B, L, D = x.shape

        # 1. Implicit RAG
        x = self.knowledge(x)

        # 2. Hierarchical abstraction
        x = self.pyramid(x)

        # 3. Working memory
        wm_out = self.working(x, write_signal=write_signal)
        x = x + wm_out

        # 4. Episodic memory – use mean-pooled state as controller
        state = x.mean(dim=1)                                           # (B, D)
        ep_out = self.episodic(state, do_write=True)                    # (B, D)
        x = x + ep_out.unsqueeze(1)

        return x
