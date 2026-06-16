"""NexusMind master configuration.

Centralises every hyperparameter across the three innovation streams:
  • Supermix  — MoE routing (v8-v14), memory (v19-v22), spiking/bio (v26),
                recursive deliberation, diffusion refinement, TTT, world-model.
  • AI-Dem-Lab — Q-learning signal router, RSI momentum, entropy-gated routing,
                 multi-agent swarm with role-based debate.
  • Xiaomi MiMo — hybrid sliding-window / global attention, multi-token
                  prediction (MTP), sparse MoE with dynamic bias routing,
                  agent post-training with preference loss.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class NexusConfig(BaseModel):
    """Master configuration for NexusMind.

    Every subsystem reads from this single source of truth so that
    experiments only require changing one YAML / dict.
    """

    # ------------------------------------------------------------------ #
    #  Model dimensions                                                    #
    # ------------------------------------------------------------------ #
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    vocab_size: int = 32000
    max_seq_len: int = 8192

    # ------------------------------------------------------------------ #
    #  MoE Configuration (from Supermix + MiMo)                            #
    # ------------------------------------------------------------------ #
    n_experts: int = 8
    n_shared_experts: int = 1
    top_k: int = 2
    routing_strategy: Literal[
        "noisy_topk",
        "hierarchical",
        "expert_choice",
        "sigma",
        "dynamic_bias",
        "multi_head_sigma",
    ] = "dynamic_bias"
    n_domain_groups: int = 2
    expert_capacity_factor: float = 1.25

    # ------------------------------------------------------------------ #
    #  Hybrid Attention (from MiMo)                                        #
    # ------------------------------------------------------------------ #
    sliding_window_size: int = 4096
    global_attention_ratio: float = 0.25  # fraction of layers → global attn

    # ------------------------------------------------------------------ #
    #  Multi-Token Prediction (from MiMo)                                  #
    # ------------------------------------------------------------------ #
    mtp_heads: int = 2
    mtp_enabled: bool = True

    # ------------------------------------------------------------------ #
    #  Thinking Engine (from Supermix recursive / deliberative)             #
    # ------------------------------------------------------------------ #
    max_thinking_steps: int = 3
    n_drafts: int = 3
    ace_threshold: float = 0.85  # Adaptive Computation Exit threshold
    enable_adversarial_debate: bool = True
    enable_diffusion_refinement: bool = False
    diffusion_steps: int = 4

    # ------------------------------------------------------------------ #
    #  Memory (from Supermix v19 / v20 / v22)                              #
    # ------------------------------------------------------------------ #
    working_memory_slots: int = 16
    episodic_memory_size: int = 64
    latent_knowledge_slots: int = 128

    # ------------------------------------------------------------------ #
    #  Spiking / Bio (from Supermix v26)                                   #
    # ------------------------------------------------------------------ #
    enable_spiking: bool = False
    spiking_timesteps: int = 4
    lif_threshold: float = 1.0
    lif_decay: float = 0.9
    liquid_tau: float = 2.0

    # ------------------------------------------------------------------ #
    #  Test-Time Training (from Supermix v26)                              #
    # ------------------------------------------------------------------ #
    enable_ttt: bool = False
    ttt_lr: float = 0.01

    # ------------------------------------------------------------------ #
    #  World Model (from Supermix v23)                                     #
    # ------------------------------------------------------------------ #
    enable_world_model: bool = False

    # ------------------------------------------------------------------ #
    #  Signals (from AI-Dem-Lab)                                           #
    # ------------------------------------------------------------------ #
    q_learning_rate: float = 0.2
    q_discount: float = 0.95
    q_epsilon: float = 0.2
    rsi_window: int = 14

    # ------------------------------------------------------------------ #
    #  Agent Swarm (from AI-Dem-Lab)                                       #
    # ------------------------------------------------------------------ #
    swarm_agents: int = 5
    agent_roles: list[str] = Field(
        default_factory=lambda: [
            "generator",
            "critic",
            "skeptic",
            "archivist",
            "anomaly_hunter",
        ]
    )

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #
    dropout: float = 0.1
    label_smoothing: float = 0.05
    preference_loss_weight: float = 0.15
    aux_loss_weight: float = 0.01
    bias_update_rate: float = 0.001
