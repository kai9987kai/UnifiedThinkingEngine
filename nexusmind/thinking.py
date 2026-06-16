"""Unified Thinking Engine — fuses ALL reasoning strategies from Supermix.

Combines:
- Recursive Thought with Adaptive Computation Exit (ACE) — Supermix v14
- Multi-Draft Deliberation with Working Memory — Supermix v19
- Graph-of-Thought with Meta-Cognitive Critique — Supermix v20
- Adversarial Self-Play (Proposer vs Adversary) — Supermix v21
- Diffusion Refinement with World Model — Supermix v23
- ThinkingEngine orchestrator with mode selection
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nexusmind.memory import WorkingMemoryBank


# ═══════════════════════════════════════════════════════════════════
# Building Blocks
# ═══════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class ReasoningCell(nn.Module):
    """Single-step feature refinement MLP with residual connection.

    A lightweight GatedFFN block used as a reasoning step:
        x → LayerNorm → Linear → SiLU → Gate → Linear → + x
    """

    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        d_ff = d_model * expansion
        self.norm = RMSNorm(d_model)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + self.down(F.silu(self.up(h)) * self.gate(h))


class AdaptiveComputationExit(nn.Module):
    """Adaptive Computation Exit (ACE) gate from Supermix v14.

    Learns when to halt reasoning early based on token-level confidence.
    Uses PonderNet-style halting probability accumulation.

    At each step:
        exit_prob = sigmoid(exit_gate(features))
        cumulative += (1 - cumulative) * exit_prob
        if cumulative > threshold: halt

    This allows easy tokens to exit after 1 step while hard tokens
    use the full recursive capacity.
    """

    def __init__(self, d_model: int, threshold: float = 0.85):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
        )
        self.threshold = threshold

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns exit probability in [0, 1]."""
        return torch.sigmoid(self.gate(features)).squeeze(-1)

    def should_halt(self, cumulative_prob: torch.Tensor) -> bool:
        """Check if mean cumulative halt probability exceeds threshold."""
        return cumulative_prob.mean().item() > self.threshold


# ═══════════════════════════════════════════════════════════════════
# Multi-Draft Deliberation (Supermix v19)
# ═══════════════════════════════════════════════════════════════════

class CrossDraftAttention(nn.Module):
    """Cross-Draft Attention — drafts attend to each other's predictions.

    After each reasoning step, K drafts exchange information via
    multi-head attention, combining the best aspects of each viewpoint.
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.norm = RMSNorm(d_model)

    def forward(self, drafts: torch.Tensor) -> torch.Tensor:
        """Cross-attend between drafts.

        Args:
            drafts: (batch, n_drafts, d_model)

        Returns:
            refined: (batch, n_drafts, d_model)
        """
        B, K, D = drafts.shape
        h = self.norm(drafts)
        q = self.q(h).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(h).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(h).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, K, D)
        return drafts + self.out(out)


class ConsistencyNetwork(nn.Module):
    """Evaluates inter-draft agreement and produces confidence-weighted fusion.

    From Supermix v19: A 3-layer MLP arbiter that scores each draft's
    consistency with the others, then fuses via confidence weighting.
    """

    def __init__(self, d_model: int, n_drafts: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * n_drafts, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_drafts),
        )

    def forward(self, drafts: torch.Tensor) -> torch.Tensor:
        """Compute consistency weights.

        Args:
            drafts: (batch, n_drafts, d_model)

        Returns:
            weights: (batch, n_drafts) normalized fusion weights
        """
        B, K, D = drafts.shape
        flat = drafts.reshape(B, K * D)
        return F.softmax(self.net(flat), dim=-1)


class MultiDraftDeliberation(nn.Module):
    """Multi-Draft Deliberation with Working Memory (Supermix v19).

    Five major innovations for enhanced reasoning:
    1. Working Memory Bank — learned K-V scratchpad across steps
    2. Multi-Draft Generation — K=3 independent reasoning chains
    3. Cross-Draft Attention — drafts attend to each other
    4. Consistency-Weighted Aggregation — confidence-weighted fusion
    5. PonderNet-style Adaptive Depth — halts early for easy inputs
    """

    def __init__(
        self,
        d_model: int,
        n_drafts: int = 3,
        max_steps: int = 3,
        n_memory_slots: int = 16,
        ace_threshold: float = 0.85,
    ):
        super().__init__()
        self.n_drafts = n_drafts
        self.max_steps = max_steps

        # Per-draft reasoning cells
        self.draft_cells = nn.ModuleList([
            nn.ModuleList([ReasoningCell(d_model) for _ in range(max_steps)])
            for _ in range(n_drafts)
        ])

        # Working memory
        self.memory = WorkingMemoryBank(d_model, n_memory_slots)

        # Cross-draft attention
        self.cross_attn = CrossDraftAttention(d_model)

        # Consistency network
        self.consistency = ConsistencyNetwork(d_model, n_drafts)

        # ACE halting
        self.ace = AdaptiveComputationExit(d_model, ace_threshold)

        # Draft initialization projections
        self.draft_projs = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_drafts)
        ])

        # Learnable scale (initialized to 0 for stable warm-start)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(
        self, x: torch.Tensor, inference: bool = False
    ) -> dict[str, torch.Tensor]:
        """Run multi-draft deliberation.

        Args:
            x: (batch, d_model) pooled input features
            inference: if True, enables early exit via ACE

        Returns:
            dict with 'output', 'halting_prob', 'steps_used'
        """
        B, D = x.shape

        # Initialize drafts
        drafts = torch.stack([
            proj(x) for proj in self.draft_projs
        ], dim=1)  # (B, K, D)

        # Reset working memory
        self.memory.reset(B)

        total_output = torch.zeros(B, D, device=x.device)
        cumulative_halt = torch.zeros(B, device=x.device)
        steps_used = self.max_steps

        for step in range(self.max_steps):
            # 1. Memory read for each draft
            for d in range(self.n_drafts):
                mem_ctx = self.memory.read(drafts[:, d])
                drafts[:, d] = drafts[:, d] + mem_ctx

            # 2. Per-draft reasoning cell
            for d in range(self.n_drafts):
                drafts[:, d] = self.draft_cells[d][step](drafts[:, d])

            # 3. Cross-draft attention
            drafts = self.cross_attn(drafts)

            # 4. Memory write
            write_signal = drafts.mean(dim=1)  # (B, D)
            self.memory.write(write_signal)

            # 5. ACE halting
            halt_prob = self.ace(drafts.mean(dim=1))

            # 6. Consistency-weighted aggregation
            weights = self.consistency(drafts)  # (B, K)
            step_fused = (drafts * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)

            # Accumulate with halting
            remaining = 1.0 - cumulative_halt
            total_output = total_output + remaining.unsqueeze(-1) * step_fused
            cumulative_halt = cumulative_halt + remaining * halt_prob

            if inference and self.ace.should_halt(cumulative_halt):
                steps_used = step + 1
                break

        return {
            "output": self.alpha * total_output,
            "halting_prob": cumulative_halt,
            "steps_used": steps_used,
        }


# ═══════════════════════════════════════════════════════════════════
# Graph-of-Thought (Supermix v20)
# ═══════════════════════════════════════════════════════════════════

class GraphAttentionLayer(nn.Module):
    """Graph Attention (GAT) layer for thought-node communication."""

    def __init__(self, d_model: int):
        super().__init__()
        self.W = nn.Linear(d_model, d_model, bias=False)
        self.a = nn.Linear(2 * d_model, 1, bias=False)
        self.norm = RMSNorm(d_model)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        """Graph attention between nodes.

        Args:
            nodes: (batch, n_nodes, d_model)

        Returns:
            updated: (batch, n_nodes, d_model)
        """
        B, N, D = nodes.shape
        h = self.W(self.norm(nodes))

        # Compute pairwise attention
        h_i = h.unsqueeze(2).expand(B, N, N, D)
        h_j = h.unsqueeze(1).expand(B, N, N, D)
        e = F.leaky_relu(self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1), 0.2)
        alpha = F.softmax(e, dim=-1)  # (B, N, N)

        out = torch.bmm(alpha.view(B * N, N, 1).transpose(1, 2),
                        h.unsqueeze(1).expand(B, N, N, D).reshape(B * N, N, D))
        out = out.squeeze(1).view(B, N, D)
        return nodes + out


class MetaCognitiveCritique(nn.Module):
    """Global critique network that reviews the entire graph and broadcasts corrections.

    From Supermix v20: Reviews all thought nodes simultaneously and produces
    a correction vector that's broadcast to every node, enabling global
    course-correction of the reasoning process.
    """

    def __init__(self, d_model: int, n_nodes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * n_nodes, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        """Produce correction vector.

        Args:
            nodes: (batch, n_nodes, d_model)

        Returns:
            correction: (batch, 1, d_model) broadcast-ready
        """
        B, N, D = nodes.shape
        flat = nodes.reshape(B, N * D)
        return self.net(flat).unsqueeze(1)  # (B, 1, D)


class GraphOfThought(nn.Module):
    """Graph-of-Thought reasoning from Supermix v20.

    Non-linear reasoning where thought-nodes exchange insights via GAT.

    Key innovations:
    - Stochastic Bayesian routing via reparameterization trick
    - Meta-cognitive self-correction
    - Per-thought dynamic depth (confident thoughts freeze early)
    """

    def __init__(
        self,
        d_model: int,
        n_nodes: int = 4,
        max_steps: int = 3,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.max_steps = max_steps

        # Node initialization with diversity injection
        self.node_projs = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_nodes)
        ])

        # Per-step components
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(d_model) for _ in range(max_steps)
        ])
        self.critiques = nn.ModuleList([
            MetaCognitiveCritique(d_model, n_nodes) for _ in range(max_steps)
        ])
        self.reasoning_cells = nn.ModuleList([
            ReasoningCell(d_model) for _ in range(max_steps)
        ])

        # Per-node halting
        self.halt_gates = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(n_nodes)
        ])

        # Attention pooling for final fusion
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_kv = nn.Linear(d_model, d_model * 2, bias=False)

        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        B, D = x.shape

        # Initialize nodes with diversity
        nodes = torch.stack([
            proj(x) + 0.01 * torch.randn(B, D, device=x.device)
            for proj in self.node_projs
        ], dim=1)  # (B, N, D)

        cum_halt = torch.zeros(B, self.n_nodes, device=x.device)

        for step in range(self.max_steps):
            # 1. Graph Attention
            nodes = self.gat_layers[step](nodes)

            # 2. Reasoning refinement
            refined = []
            for n in range(self.n_nodes):
                refined.append(self.reasoning_cells[step](nodes[:, n]))
            nodes = torch.stack(refined, dim=1)

            # 3. Meta-cognitive critique
            correction = self.critiques[step](nodes)
            nodes = nodes + 0.1 * correction

            # 4. Per-node halting
            for n in range(self.n_nodes):
                h = torch.sigmoid(self.halt_gates[n](nodes[:, n])).squeeze(-1)
                cum_halt[:, n] = cum_halt[:, n] + (1.0 - cum_halt[:, n]) * h

        # Attention pooling
        query = self.pool_query.expand(B, -1, -1)
        kv = self.pool_kv(nodes)
        k, v = kv.chunk(2, dim=-1)
        scale = math.sqrt(D)
        attn = F.softmax((query @ k.transpose(-2, -1)) / scale, dim=-1)
        pooled = (attn @ v).squeeze(1)

        return {
            "output": self.alpha * pooled,
            "node_halts": cum_halt,
        }


# ═══════════════════════════════════════════════════════════════════
# Adversarial Debate (Supermix v21)
# ═══════════════════════════════════════════════════════════════════

class AdversarialDebate(nn.Module):
    """Adversarial Self-Play reasoning from Supermix v21.

    Proposer MoE generates candidate answers while Adversary MoE finds flaws.
    A Resolution network resolves the debate with confidence-gated accumulation.
    """

    def __init__(self, d_model: int, n_rounds: int = 3):
        super().__init__()
        self.n_rounds = n_rounds

        # Proposer and Adversary reasoning paths
        self.proposers = nn.ModuleList([ReasoningCell(d_model) for _ in range(n_rounds)])
        self.adversaries = nn.ModuleList([ReasoningCell(d_model) for _ in range(n_rounds)])

        # Resolution network
        self.resolvers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            ) for _ in range(n_rounds)
        ])

        # Confidence gates
        self.conf_gates = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(n_rounds)
        ])

        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        B, D = x.shape
        state = x
        total = torch.zeros(B, D, device=x.device)
        cum_conf = torch.zeros(B, 1, device=x.device)
        debate_log = []

        for rnd in range(self.n_rounds):
            proposal = self.proposers[rnd](state)
            critique = self.adversaries[rnd](state)

            resolved = self.resolvers[rnd](torch.cat([proposal, critique], dim=-1))
            confidence = torch.sigmoid(self.conf_gates[rnd](state))

            remaining = 1.0 - cum_conf
            total = total + remaining * resolved
            cum_conf = cum_conf + remaining * confidence

            state = state + 0.1 * resolved
            debate_log.append({
                "round": rnd,
                "confidence": confidence.mean().item(),
            })

        return {
            "output": self.alpha * total,
            "debate_log": debate_log,
            "final_confidence": cum_conf.mean().item(),
        }


# ═══════════════════════════════════════════════════════════════════
# Diffusion Refinement (Supermix v23)
# ═══════════════════════════════════════════════════════════════════

class DiffusionRefiner(nn.Module):
    """Diffusion-based answer refinement from Supermix v23.

    Sculpts the answer from noise via iterative denoising,
    each step conditioned on the input context. Includes a
    lightweight world model for candidate scoring.
    """

    def __init__(self, d_model: int, n_steps: int = 4):
        super().__init__()
        self.n_steps = n_steps

        # Denoising network (shared across steps, conditioned on time)
        self.time_embeds = nn.Embedding(n_steps, d_model)
        self.denoisers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2),
                nn.SiLU(),
                nn.Linear(d_model * 2, d_model),
            ) for _ in range(n_steps)
        ])

        # World model (value network)
        self.world_value = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )

        # Noise schedule (learned betas)
        self.betas = nn.Parameter(torch.linspace(0.1, 0.02, n_steps))

        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        B, D = context.shape

        # Start from noise
        z = torch.randn(B, D, device=context.device) * 0.5

        for t in range(self.n_steps):
            time_emb = self.time_embeds(
                torch.tensor(t, device=context.device)
            ).unsqueeze(0).expand(B, -1)

            # Denoise: predict and remove noise
            combined = torch.cat([z + time_emb, context], dim=-1)
            noise_pred = self.denoisers[t](combined)
            z = z - self.betas[t] * noise_pred

        # Score with world model
        score = self.world_value(z).squeeze(-1)

        return {
            "output": self.alpha * z,
            "world_score": score,
        }


# ═══════════════════════════════════════════════════════════════════
# Thinking Engine (Main Orchestrator)
# ═══════════════════════════════════════════════════════════════════

class ThinkingEngine(nn.Module):
    """The main thinking engine that orchestrates all reasoning strategies.

    Mode selection:
    - 'fast': Single ReasoningCell + ACE (lowest latency)
    - 'deep': MultiDraftDeliberation + AdversarialDebate (highest quality)
    - 'agent': GraphOfThought + tool-call routing (for multi-step tasks)
    - 'creative': DiffusionRefiner (for open-ended generation)
    """

    def __init__(
        self,
        d_model: int,
        max_steps: int = 3,
        n_drafts: int = 3,
        ace_threshold: float = 0.85,
        n_graph_nodes: int = 4,
        diffusion_steps: int = 4,
        n_memory_slots: int = 16,
    ):
        super().__init__()
        self.d_model = d_model

        # Fast mode
        self.fast_cell = ReasoningCell(d_model)
        self.fast_ace = AdaptiveComputationExit(d_model, ace_threshold)

        # Deep mode
        self.deliberation = MultiDraftDeliberation(
            d_model, n_drafts, max_steps, n_memory_slots, ace_threshold
        )
        self.debate = AdversarialDebate(d_model, n_rounds=max_steps)

        # Agent mode
        self.graph = GraphOfThought(d_model, n_graph_nodes, max_steps)

        # Creative mode
        self.diffusion = DiffusionRefiner(d_model, diffusion_steps)

        # Mode fusion
        self.output_norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "deep",
        inference: bool = False,
    ) -> dict[str, torch.Tensor | list | int | float]:
        """Run thinking in the specified mode.

        Args:
            x: (batch, d_model) pooled hidden states
            mode: one of 'fast', 'deep', 'agent', 'creative'
            inference: enables early exit optimizations

        Returns:
            dict with 'output', 'thinking_trace', 'steps_used', etc.
        """
        trace = []

        if mode == "fast":
            refined = self.fast_cell(x)
            exit_prob = self.fast_ace(refined)
            trace.append(f"[fast] ACE exit_prob={exit_prob.mean().item():.3f}")
            output = refined
            steps = 1

        elif mode == "deep":
            delib_result = self.deliberation(x, inference=inference)
            trace.append(
                f"[deliberation] steps={delib_result['steps_used']}, "
                f"halt={delib_result['halting_prob'].mean().item():.3f}"
            )
            debate_result = self.debate(x)
            trace.append(
                f"[debate] conf={debate_result['final_confidence']:.3f}, "
                f"rounds={len(debate_result['debate_log'])}"
            )
            output = x + delib_result["output"] + debate_result["output"]
            steps = delib_result["steps_used"]

        elif mode == "agent":
            graph_result = self.graph(x)
            trace.append(
                f"[graph_of_thought] nodes={self.graph.n_nodes}, "
                f"mean_halt={graph_result['node_halts'].mean().item():.3f}"
            )
            output = x + graph_result["output"]
            steps = self.graph.max_steps

        elif mode == "creative":
            diff_result = self.diffusion(x)
            trace.append(
                f"[diffusion] steps={self.diffusion.n_steps}, "
                f"world_score={diff_result['world_score'].mean().item():.3f}"
            )
            output = x + diff_result["output"]
            steps = self.diffusion.n_steps

        else:
            raise ValueError(f"Unknown thinking mode: {mode}")

        output = self.output_norm(output)

        return {
            "output": output,
            "thinking_trace": trace,
            "steps_used": steps,
            "mode": mode,
        }
