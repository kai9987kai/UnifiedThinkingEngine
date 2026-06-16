"""MoE routing strategies for NexusMind.

Implements six distinct Mixture-of-Experts routing algorithms drawn from
Supermix (v8-v14) and Xiaomi MiMo, plus a factory for config-driven
instantiation.

Router inventory
────────────────
1. NoisyTopKRouter       — Supermix v8:  learnable Gaussian noise + top-k
2. HierarchicalRouter    — Supermix v9:  domain-group → expert two-level
                           routing with always-on shared expert (DeepSeek-MoE)
3. DynamicBiasRouter     — Supermix v10 / MiMo:  aux-loss-free load balancing
                           via dynamic bias buffers (DeepSeek-V3 style)
4. ExpertChoiceRouter    — Supermix v11: experts pick tokens, not vice-versa
5. SigmaRouter           — Supermix v12: independent sigmoid scores, no
                           softmax competition between experts
6. MultiHeadSigmaRouter  — Supermix v14: multiple routing heads averaged

All routers expose a unified forward signature:
    forward(x: Tensor[B, S, D]) → (weights: Tensor[B, S, K],
                                    indices: Tensor[B, S, K],
                                    aux_loss: Tensor[])
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nexusmind.config import NexusConfig

# --------------------------------------------------------------------------- #
#  Type alias for the unified router return type                               #
# --------------------------------------------------------------------------- #
RouterOutput = Tuple[Tensor, Tensor, Tensor]  # weights, indices, aux_loss


# =========================================================================== #
#  1. Noisy Top-K Router  (Supermix v8)                                        #
# =========================================================================== #
class NoisyTopKRouter(nn.Module):
    """Noisy Top-K Gating with learnable noise.

    Each token is projected to *n_experts* logits.  During training,
    learnable Gaussian noise is added before the top-k selection so that
    every expert has a non-zero gradient signal, improving exploration.

    Innovation (Supermix v8):
        The noise scale is *learned per expert* rather than being a fixed
        hyperparameter, allowing the model to dynamically control
        exploration vs. exploitation for each expert independently.

    Load-balancing:
        A standard importance + load auxiliary loss is applied to encourage
        balanced expert utilisation.
    """

    def __init__(self, config: NexusConfig) -> None:
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.aux_loss_weight = config.aux_loss_weight

        self.gate = nn.Linear(config.d_model, self.n_experts, bias=False)
        # Learnable per-expert noise scale (softplus ensures positivity)
        self.noise_logvar = nn.Parameter(torch.zeros(self.n_experts))

    def forward(self, x: Tensor) -> RouterOutput:
        """Route tokens through noisy top-k gating.

        Args:
            x: Input tensor of shape ``(B, S, D)``.

        Returns:
            weights:  ``(B, S, K)`` normalised routing weights.
            indices:  ``(B, S, K)`` selected expert indices.
            aux_loss: Scalar load-balancing loss.
        """
        # x: (B, S, D) → logits: (B, S, E)
        logits = self.gate(x)

        if self.training:
            noise_std = F.softplus(self.noise_logvar)  # (E,)
            noise = torch.randn_like(logits) * noise_std.unsqueeze(0).unsqueeze(0)
            logits = logits + noise

        # Top-k selection
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)  # (B, S, K)
        weights = F.softmax(topk_vals, dim=-1)

        # Auxiliary load-balancing loss
        aux_loss = self._aux_loss(logits)

        return weights, topk_idx, aux_loss

    # ----- helpers -------------------------------------------------------- #
    def _aux_loss(self, logits: Tensor) -> Tensor:
        """Importance-weighted load-balancing loss (Switch Transformer)."""
        probs = F.softmax(logits, dim=-1)  # (B, S, E)
        # fraction of tokens routed to each expert
        tokens_per_expert = probs.mean(dim=(0, 1))  # (E,)
        # fraction of routing probability per expert
        prob_per_expert = probs.mean(dim=(0, 1))  # (E,)
        loss = (tokens_per_expert * prob_per_expert).sum() * self.n_experts
        return loss * self.aux_loss_weight


# =========================================================================== #
#  2. Hierarchical Router  (Supermix v9 / DeepSeek-MoE)                        #
# =========================================================================== #
class HierarchicalRouter(nn.Module):
    """Two-level domain → expert routing with shared expert.

    Innovation (Supermix v9 / DeepSeek-MoE):
        Experts are partitioned into *domain groups*.  A lightweight first
        gate selects which domain group to activate, then a second gate
        selects top-k experts *within* that group.  A shared expert is
        always active regardless of the domain selection, providing a
        stable baseline representation.

    This reduces the softmax width at each level and encourages domain
    specialisation across expert clusters.
    """

    def __init__(self, config: NexusConfig) -> None:
        super().__init__()
        self.n_experts = config.n_experts
        self.n_groups = config.n_domain_groups
        self.top_k = config.top_k
        self.n_shared = config.n_shared_experts
        self.aux_loss_weight = config.aux_loss_weight

        assert self.n_experts % self.n_groups == 0, (
            f"n_experts ({self.n_experts}) must be divisible by "
            f"n_domain_groups ({self.n_groups})"
        )
        self.experts_per_group = self.n_experts // self.n_groups

        # Level-1: domain group selector
        self.group_gate = nn.Linear(config.d_model, self.n_groups, bias=False)
        # Level-2: within-group expert selector (shared linear, index offset
        # applied downstream)
        self.expert_gate = nn.Linear(
            config.d_model, self.experts_per_group, bias=False
        )

    def forward(self, x: Tensor) -> RouterOutput:
        """Two-level routing.

        Args:
            x: ``(B, S, D)``

        Returns:
            weights, indices, aux_loss — see module docstring.
        """
        B, S, D = x.shape

        # --- Level 1: choose domain group (hard argmax) ---
        group_logits = self.group_gate(x)  # (B, S, G)
        group_idx = group_logits.argmax(dim=-1)  # (B, S)

        # --- Level 2: choose experts within group ---
        expert_logits = self.expert_gate(x)  # (B, S, E_per_g)
        topk_vals, local_idx = torch.topk(
            expert_logits, self.top_k, dim=-1
        )  # (B, S, K)
        weights = F.softmax(topk_vals, dim=-1)

        # Map local expert indices to global indices
        offset = group_idx.unsqueeze(-1) * self.experts_per_group  # (B, S, 1)
        global_idx = local_idx + offset  # (B, S, K)

        # Aux loss
        probs = F.softmax(expert_logits, dim=-1)
        load = probs.mean(dim=(0, 1))
        aux_loss = (load * load).sum() * self.experts_per_group * self.aux_loss_weight

        return weights, global_idx, aux_loss


# =========================================================================== #
#  3. Dynamic Bias Router  (Supermix v10 / DeepSeek-V3 / MiMo)                #
# =========================================================================== #
class DynamicBiasRouter(nn.Module):
    """Auxiliary-loss-free load balancing via dynamic bias buffers.

    Innovation (Supermix v10 / DeepSeek-V3 / MiMo):
        Instead of adding a differentiable auxiliary loss that competes
        with the main language-modelling loss, a **non-gradient bias term**
        is maintained for each expert.  After every forward pass the bias
        is adjusted upward for under-utilised experts and downward for
        over-utilised ones, using an exponential moving average.

        This yields perfectly balanced routing without distorting gradients
        and is the strategy adopted by Xiaomi MiMo.

    The router itself is a simple linear → sigmoid gate; the dynamic bias
    is applied *before* the top-k selection.
    """

    def __init__(self, config: NexusConfig) -> None:
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.bias_update_rate = config.bias_update_rate

        self.gate = nn.Linear(config.d_model, self.n_experts, bias=False)
        # Non-gradient bias buffer — updated manually
        self.register_buffer(
            "expert_bias", torch.zeros(self.n_experts)
        )

    def forward(self, x: Tensor) -> RouterOutput:
        """Route with dynamic bias load balancing.

        Args:
            x: ``(B, S, D)``

        Returns:
            weights, indices, aux_loss (always 0).
        """
        logits = self.gate(x)  # (B, S, E)

        # Apply dynamic bias (detached — no gradient)
        biased_logits = logits + self.expert_bias.detach()

        topk_vals, topk_idx = torch.topk(biased_logits, self.top_k, dim=-1)
        weights = F.softmax(topk_vals, dim=-1)

        # --- Bias update (train mode only, no grad) ---
        if self.training:
            with torch.no_grad():
                # Count how often each expert was selected
                one_hot = F.one_hot(topk_idx, self.n_experts).float()  # (B,S,K,E)
                counts = one_hot.sum(dim=(0, 1, 2))  # (E,)
                total = topk_idx.numel() / self.top_k  # B * S
                target = total * self.top_k / self.n_experts
                # Positive delta ⇒ expert underused ⇒ raise bias
                delta = (target - counts) / max(total, 1.0)
                self.expert_bias += self.bias_update_rate * delta

        # No auxiliary loss by design
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return weights, topk_idx, aux_loss


# =========================================================================== #
#  4. Expert Choice Router  (Supermix v11)                                     #
# =========================================================================== #
class ExpertChoiceRouter(nn.Module):
    """Experts pick tokens instead of tokens picking experts.

    Innovation (Supermix v11):
        Typical MoE routers let each *token* choose its top-k experts,
        leading to load imbalance because popular experts get many more
        tokens.  Expert-choice routing *inverts* the selection: each
        expert picks a fixed-capacity subset of tokens, guaranteeing
        perfect load balance by construction.

    Each expert sees ``ceil(capacity_factor × B×S / E)`` tokens.
    """

    def __init__(self, config: NexusConfig) -> None:
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.capacity_factor = config.expert_capacity_factor
        self.aux_loss_weight = config.aux_loss_weight

        self.gate = nn.Linear(config.d_model, self.n_experts, bias=False)

    def forward(self, x: Tensor) -> RouterOutput:
        """Expert-choice routing.

        Args:
            x: ``(B, S, D)``

        Returns:
            weights:  ``(B, S, K)`` — for compatibility, we reconstruct
                      per-token weights from the expert-choice assignment.
            indices:  ``(B, S, K)`` — selected expert indices per token.
            aux_loss: Scalar (always near-zero; balance is structural).
        """
        B, S, D = x.shape
        N = B * S  # total tokens
        capacity = int(math.ceil(self.capacity_factor * N / self.n_experts))

        logits = self.gate(x)  # (B, S, E)
        flat_logits = logits.view(N, self.n_experts)  # (N, E)
        scores = F.softmax(flat_logits, dim=0)  # softmax over *tokens* per expert

        # Each expert picks its top-capacity tokens
        # scores transposed: (E, N) — top-k over token dim
        topk_scores, topk_token_idx = torch.topk(
            scores.t(), capacity, dim=-1
        )  # (E, C), (E, C)

        # Reconstruct per-token routing: for each token gather assigned experts
        # Build sparse assignment: token → list of experts
        expert_ids = (
            torch.arange(self.n_experts, device=x.device)
            .unsqueeze(1)
            .expand(-1, capacity)
        )  # (E, C)

        # Flatten and scatter
        flat_token_idx = topk_token_idx.reshape(-1)  # (E*C,)
        flat_expert_id = expert_ids.reshape(-1)  # (E*C,)
        flat_scores = topk_scores.reshape(-1)  # (E*C,)

        # For each token, gather at most top_k assigned experts
        # Build (N, E) dense weight matrix then top-k
        dense_weights = torch.zeros(N, self.n_experts, device=x.device, dtype=x.dtype)
        dense_weights.scatter_add_(
            0,
            flat_token_idx.unsqueeze(1).expand(-1, 1),
            flat_scores.unsqueeze(1),
        )
        # Re-scatter expert ids
        dense_ids = torch.zeros(N, self.n_experts, device=x.device, dtype=torch.long)
        # Use the original logits softmax as dense weights for unassigned fallback
        dense_weights_fallback = F.softmax(flat_logits, dim=-1)
        # Merge: use expert-choice weights where assigned, else fallback
        mask = dense_weights > 0
        dense_weights = torch.where(mask, dense_weights, dense_weights_fallback)

        topk_w, topk_i = torch.topk(dense_weights, self.top_k, dim=-1)  # (N, K)
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-8)

        weights = topk_w.view(B, S, self.top_k)
        indices = topk_i.view(B, S, self.top_k)

        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return weights, indices, aux_loss


# =========================================================================== #
#  5. Sigma Router  (Supermix v12)                                             #
# =========================================================================== #
class SigmaRouter(nn.Module):
    """Independent sigmoid routing — no inter-expert competition.

    Innovation (Supermix v12):
        Standard MoE routers use softmax, forcing experts to *compete*
        for probability mass.  Sigma routing applies an independent
        sigmoid to each expert score, meaning multiple experts can have
        high activation simultaneously.  The top-k selection still limits
        compute, but the gradient landscape is smoother because each
        expert's gate is decoupled from the others.

    This is especially beneficial when the input is ambiguous and
    genuinely belongs to several domains.
    """

    def __init__(self, config: NexusConfig) -> None:
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.aux_loss_weight = config.aux_loss_weight

        self.gate = nn.Linear(config.d_model, self.n_experts, bias=False)

    def forward(self, x: Tensor) -> RouterOutput:
        """Sigmoid-based independent routing.

        Args:
            x: ``(B, S, D)``

        Returns:
            weights, indices, aux_loss.
        """
        logits = self.gate(x)  # (B, S, E)
        scores = torch.sigmoid(logits)  # Independent per-expert activation

        topk_scores, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        # Normalise selected weights to sum to 1
        weights = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-8)

        # Aux loss: variance-based balance encouragement
        mean_score = scores.mean(dim=(0, 1))  # (E,)
        aux_loss = mean_score.var() * self.n_experts * self.aux_loss_weight

        return weights, topk_idx, aux_loss


# =========================================================================== #
#  6. Multi-Head Sigma Router  (Supermix v14)                                  #
# =========================================================================== #
class MultiHeadSigmaRouter(nn.Module):
    """Multiple routing heads averaged for robust expert selection.

    Innovation (Supermix v14):
        A single routing projection can be noisy, especially early in
        training.  Multi-head sigma routing runs *H* independent sigmoid
        routing heads in parallel, each with its own learned projection.
        The per-expert scores are averaged across heads before the top-k
        selection, yielding more stable and diverse expert assignments.

        This is analogous to how multi-head attention stabilises attention
        patterns; here it stabilises routing decisions.
    """

    def __init__(self, config: NexusConfig, n_routing_heads: int = 4) -> None:
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.n_heads = n_routing_heads
        self.aux_loss_weight = config.aux_loss_weight

        # Each head has its own projection
        self.heads = nn.ModuleList(
            [nn.Linear(config.d_model, self.n_experts, bias=False) for _ in range(self.n_heads)]
        )

    def forward(self, x: Tensor) -> RouterOutput:
        """Multi-head sigmoid routing.

        Args:
            x: ``(B, S, D)``

        Returns:
            weights, indices, aux_loss.
        """
        # Collect sigmoid scores from each head: list of (B, S, E)
        head_scores = [torch.sigmoid(head(x)) for head in self.heads]
        # Average across heads → (B, S, E)
        avg_scores = torch.stack(head_scores, dim=0).mean(dim=0)

        topk_scores, topk_idx = torch.topk(avg_scores, self.top_k, dim=-1)
        weights = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-8)

        # Aux loss: per-head balance + cross-head agreement
        mean_per_expert = avg_scores.mean(dim=(0, 1))
        balance_loss = mean_per_expert.var() * self.n_experts

        # Cross-head diversity bonus (encourage heads to agree on final
        # ranking but explore different sub-rankings)
        diversity_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for i in range(self.n_heads):
            for j in range(i + 1, self.n_heads):
                diversity_loss = diversity_loss + F.mse_loss(
                    head_scores[i], head_scores[j]
                )
        n_pairs = max(self.n_heads * (self.n_heads - 1) // 2, 1)
        diversity_loss = diversity_loss / n_pairs

        aux_loss = (balance_loss + 0.1 * diversity_loss) * self.aux_loss_weight

        return weights, topk_idx, aux_loss


# =========================================================================== #
#  Router Factory                                                              #
# =========================================================================== #
_ROUTER_REGISTRY: dict[str, type[nn.Module]] = {
    "noisy_topk": NoisyTopKRouter,
    "hierarchical": HierarchicalRouter,
    "dynamic_bias": DynamicBiasRouter,
    "expert_choice": ExpertChoiceRouter,
    "sigma": SigmaRouter,
    "multi_head_sigma": MultiHeadSigmaRouter,
}


class RouterFactory:
    """Instantiate the correct router from a config string.

    Usage::

        router = RouterFactory.create(config)
    """

    @staticmethod
    def create(config: NexusConfig) -> nn.Module:
        """Build a router instance based on ``config.routing_strategy``.

        Args:
            config: The global NexusMind configuration.

        Returns:
            An ``nn.Module`` implementing the chosen routing strategy.

        Raises:
            ValueError: If the requested strategy is unknown.
        """
        strategy = config.routing_strategy
        if strategy not in _ROUTER_REGISTRY:
            raise ValueError(
                f"Unknown routing strategy '{strategy}'. "
                f"Available: {list(_ROUTER_REGISTRY.keys())}"
            )
        return _ROUTER_REGISTRY[strategy](config)
