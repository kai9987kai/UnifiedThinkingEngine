"""NexusMind Core — The Unified Hybrid Thinking Model.

Brings together ALL modules into a single cohesive transformer:
- Hybrid Attention (sliding-window + global) from Xiaomi MiMo
- Sparse MoE with 6 routing strategies from Supermix
- Multi-Token Prediction from MiMo
- Thinking Engine (recursive thought, deliberation, debate, diffusion) from Supermix
- Working Memory, Episodic Memory, Latent Knowledge Core from Supermix v19-v22
- Signal intelligence (Q-learning, RSI) from AI-Dem-Lab
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nexusmind.config import NexusConfig
from nexusmind.attention import HybridAttention
from nexusmind.routing import RouterFactory
from nexusmind.memory import WorkingMemoryBank, EpisodicMemory, LatentKnowledgeCore
from nexusmind.thinking import ThinkingEngine, RMSNorm
from nexusmind.mtp import MultiTokenPredictionHead


# ═══════════════════════════════════════════════════════════════════
# Expert Feed-Forward Network
# ═══════════════════════════════════════════════════════════════════

class ExpertFFN(nn.Module):
    """Single expert feed-forward network with configurable activation.

    From Supermix: each expert uses a different activation function
    (SiLU, GELU, Mish, ReLU, SELU, Tanh) for representational diversity.
    Research shows homogeneous experts converge to similar representations.
    """

    ACTIVATIONS = {
        0: ("SiLU", nn.SiLU()),
        1: ("GELU", nn.GELU()),
        2: ("Mish", nn.Mish()),
        3: ("ReLU", nn.ReLU()),
        4: ("SELU", nn.SELU()),
        5: ("Tanh", nn.Tanh()),
        6: ("SiLU", nn.SiLU()),
        7: ("GELU", nn.GELU()),
    }

    def __init__(self, d_model: int, d_ff: int, expert_idx: int = 0):
        super().__init__()
        _, act = self.ACTIVATIONS.get(expert_idx % len(self.ACTIVATIONS), ("SiLU", nn.SiLU()))
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.up(x)) * self.gate(x))


# ═══════════════════════════════════════════════════════════════════
# MoE Layer
# ═══════════════════════════════════════════════════════════════════

class MoELayer(nn.Module):
    """Mixture-of-Experts layer with pluggable routing strategy.

    Supports all 6 routing strategies from Supermix + MiMo:
    - Noisy Top-K, Hierarchical, Dynamic Bias, Expert Choice, Sigma, Multi-Head Sigma

    Also includes a shared expert (always active) for universal knowledge,
    following the DeepSeek-MoE / MiMo pattern.
    """

    def __init__(self, config: NexusConfig, layer_idx: int = 0):
        super().__init__()
        self.d_model = config.d_model
        self.n_experts = config.n_experts

        # Routed experts with heterogeneous activations
        self.experts = nn.ModuleList([
            ExpertFFN(config.d_model, config.d_ff, expert_idx=i)
            for i in range(config.n_experts)
        ])

        # Shared expert (always-on, from DeepSeek-MoE / MiMo)
        self.shared_expert = ExpertFFN(config.d_model, config.d_ff * 2, expert_idx=0)
        self.shared_scale = nn.Parameter(torch.tensor(0.5))

        # Router (takes NexusConfig object)
        self.router = RouterFactory.create(config)

        self.norm = RMSNorm(config.d_model)

        # Aux loss accumulator
        self.last_aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MoE layer.

        Args:
            x: (batch, seq, d_model)

        Returns:
            output: (batch, seq, d_model) with residual
        """
        B, S, D = x.shape
        residual = x
        h = self.norm(x)

        # Shared expert (always active)
        shared_out = self.shared_expert(h) * self.shared_scale

        # Route through experts — router returns (weights, indices, aux_loss)
        # with shapes (B, S, K), (B, S, K), scalar
        weights, indices, aux_loss = self.router(h)
        self.last_aux_loss = aux_loss

        # Dispatch to experts
        h_flat = h.reshape(B * S, D)
        weights_flat = weights.reshape(B * S, -1)   # (B*S, K)
        indices_flat = indices.reshape(B * S, -1)    # (B*S, K)
        K = weights_flat.size(1)

        routed_out = torch.zeros(B * S, D, device=x.device, dtype=x.dtype)
        for k_idx in range(K):
            expert_idx = indices_flat[:, k_idx]                  # (B*S,)
            expert_weight = weights_flat[:, k_idx].unsqueeze(-1)  # (B*S, 1)

            for e in range(self.n_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_in = h_flat[mask]
                    expert_out = self.experts[e](expert_in)
                    routed_out[mask] += expert_weight[mask] * expert_out

        routed_out = routed_out.reshape(B, S, D)
        return residual + shared_out + routed_out


# ═══════════════════════════════════════════════════════════════════
# Transformer Block
# ═══════════════════════════════════════════════════════════════════

class NexusMindBlock(nn.Module):
    """Single transformer block with Hybrid Attention + MoE FFN.

    Architecture per block:
        x → RMSNorm → HybridAttention → + x → RMSNorm → MoE → + x

    Hybrid attention alternates between sliding-window (local) and
    global attention based on layer index, following the MiMo pattern.
    """

    def __init__(self, config: NexusConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = HybridAttention(config, layer_idx)
        self.attn_norm = RMSNorm(config.d_model)

        self.moe = MoELayer(config, layer_idx)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        h = self.attn_norm(x)
        attn_out, _ = self.attn(h)
        x = x + attn_out

        # MoE FFN (norm is inside MoELayer)
        x = self.moe(x)

        return x


# ═══════════════════════════════════════════════════════════════════
# NexusMind — The Unified Model
# ═══════════════════════════════════════════════════════════════════

class NexusMind(nn.Module):
    """NexusMind — Unified Hybrid Thinking Model.

    Fuses innovations from three projects:

    **Supermix** (v8–v26):
        - 6 MoE routing strategies with heterogeneous expert activations
        - Recursive Thought with Adaptive Computation Exit (ACE)
        - Multi-Draft Deliberation with Working Memory
        - Graph-of-Thought with Meta-Cognitive Critique
        - Adversarial Self-Play (Proposer vs Adversary)
        - Diffusion Refinement with World Model

    **Xiaomi MiMo**:
        - Hybrid Attention (sliding-window + global interleaved)
        - Multi-Token Prediction for speculative decoding
        - Sparse MoE with shared experts and dynamic bias routing

    **AI-Dem-Lab**:
        - Q-Learning policy for adaptive routing
        - RSI momentum for output stability
        - Multi-agent swarm orchestration

    Architecture:
        token_embed + pos_embed
        → N × NexusMindBlock (HybridAttn + MoE)
        → Latent Knowledge Core query
        → ThinkingEngine (mode-dependent)
        → LM head + MTP heads
    """

    def __init__(self, config: NexusConfig):
        super().__init__()
        self.config = config

        # ── Embeddings ──
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # ── Transformer Blocks ──
        self.blocks = nn.ModuleList([
            NexusMindBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # ── Memory Systems ──
        self.working_memory = WorkingMemoryBank(config.d_model, config.working_memory_slots)
        self.episodic_memory = EpisodicMemory(config.d_model, config.episodic_memory_size)
        self.knowledge_core = LatentKnowledgeCore(config.d_model, config.latent_knowledge_slots)

        # ── Thinking Engine ──
        self.thinking_engine = ThinkingEngine(
            d_model=config.d_model,
            max_steps=config.max_thinking_steps,
            n_drafts=config.n_drafts,
            ace_threshold=config.ace_threshold,
            n_graph_nodes=4,
            diffusion_steps=config.diffusion_steps,
            n_memory_slots=config.working_memory_slots,
        )

        # ── Output Heads ──
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (embedding <-> lm_head)
        self.lm_head.weight = self.token_embed.weight

        # Multi-Token Prediction (from MiMo)
        if config.mtp_enabled:
            self.mtp = MultiTokenPredictionHead(
                config.d_model, config.vocab_size, config.mtp_heads
            )
        else:
            self.mtp = None

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        thinking_mode: str = "fast",
        inference: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            input_ids: (batch, seq) token ids
            attention_mask: (batch, seq) optional mask
            target_ids: (batch, seq) for training loss computation
            thinking_mode: 'fast', 'deep', 'agent', or 'creative'
            inference: enables early exit optimizations

        Returns:
            dict with 'logits', 'loss' (if target_ids), 'mtp_logits',
            'thinking_trace', 'thinking_output'
        """
        B, S = input_ids.shape

        # 1. Embed tokens
        x = self.token_embed(input_ids)
        x = self.drop(x)

        # 2. Position IDs
        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)

        # 3. Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask, position_ids=position_ids)

        # 4. Query Latent Knowledge Core (implicit RAG over sequence)
        x = self.knowledge_core(x)
        pooled = x.mean(dim=1)  # (B, D) — simple mean pooling

        # 5. Thinking Engine
        thinking_result = self.thinking_engine(
            pooled, mode=thinking_mode, inference=inference
        )
        # Broadcast thinking output back to sequence
        thinking_broadcast = thinking_result["output"].unsqueeze(1)  # (B, 1, D)
        x = x + thinking_broadcast * 0.1

        # 6. Final norm + LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, S, V)

        result = {
            "logits": logits,
            "hidden_states": x,
            "thinking_trace": thinking_result.get("thinking_trace", []),
            "thinking_steps": thinking_result.get("steps_used", 0),
            "thinking_mode": thinking_mode,
        }

        # 7. MTP
        if self.mtp is not None:
            mtp_result = self.mtp(x, target_ids=target_ids)
            result["mtp_logits"] = mtp_result["logits"]
            result["mtp_predictions"] = mtp_result["predictions"]
            if "loss" in mtp_result:
                result["mtp_loss"] = mtp_result["loss"]

        # 8. Loss
        if target_ids is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = target_ids[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                label_smoothing=self.config.label_smoothing,
            )
            total_loss = lm_loss
            if "mtp_loss" in result:
                total_loss = total_loss + 0.3 * result["mtp_loss"]
            result["loss"] = total_loss
            result["lm_loss"] = lm_loss

        return result

    def think(
        self,
        input_ids: torch.Tensor,
        mode: str = "deep",
        max_new_tokens: int = 128,
    ) -> dict:
        """High-level thinking API for inference.

        Args:
            input_ids: (batch, seq) prompt token ids
            mode: thinking mode ('fast', 'deep', 'agent', 'creative')
            max_new_tokens: maximum tokens to generate

        Returns:
            dict with 'generated_ids', 'thinking_trace', 'thinking_steps'
        """
        self.eval()
        device = input_ids.device
        generated = input_ids.clone()
        all_traces = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Truncate context if needed
                ctx = generated[:, -self.config.max_seq_len:]
                result = self.forward(ctx, thinking_mode=mode, inference=True)

                logits = result["logits"][:, -1, :]  # (B, V)
                next_token = logits.argmax(dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=1)
                all_traces.extend(result.get("thinking_trace", []))

                # Use MTP for speculative look-ahead
                if self.mtp is not None:
                    mtp_preds = result.get("mtp_predictions", [])
                    # Could extend with speculative verification here

        return {
            "generated_ids": generated,
            "thinking_trace": all_traces,
            "thinking_steps": result.get("thinking_steps", 0),
        }

    @staticmethod
    def from_config(config: NexusConfig) -> "NexusMind":
        """Create a NexusMind model from config."""
        return NexusMind(config)

    def count_parameters(self) -> dict[str, int]:
        """Count parameters by component."""
        counts = {}
        for name, module in self.named_children():
            counts[name] = sum(p.numel() for p in module.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts
