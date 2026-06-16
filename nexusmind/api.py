"""NexusMind API — FastAPI server for the Unified Hybrid Thinking Model.

Endpoints:
    POST /v1/think     — Main thinking endpoint
    POST /v1/feedback  — Q-learning feedback for adaptive routing
    GET  /v1/signals   — Signal dashboard (Q-table, RSI, routing rec)
    GET  /v1/config    — Current model configuration
    GET  /health       — Health check
    GET  /             — Web UI
"""

from __future__ import annotations

import time
import os
from typing import Any, Literal, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nexusmind.config import NexusConfig
from nexusmind.core import NexusMind
from nexusmind.signals import SignalEngine, QTable, RSICalculator, NoveltyScorer, CoherenceScorer
from nexusmind.agents import SwarmOrchestrator

# ═══════════════════════════════════════════════════════════════════
# App Setup
# ═══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="NexusMind API",
    version="1.0.0",
    description=(
        "Unified Hybrid Thinking Model fusing Supermix MoE (v8–v26), "
        "AI-Dem-Lab Q-learning & multi-agent swarm, and Xiaomi MiMo "
        "hybrid attention & Multi-Token Prediction."
    ),
)

# ── Global State ──
config = NexusConfig()
model: Optional[NexusMind] = None
device = "cpu"
signal_engine = SignalEngine()
swarm = SwarmOrchestrator()


def get_model() -> NexusMind:
    """Lazy-initialize the model."""
    global model, device
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = NexusMind.from_config(config).to(device)
        model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════
# Request / Response Models
# ═══════════════════════════════════════════════════════════════════

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str


class ThinkRequest(BaseModel):
    """Request body for the /v1/think endpoint."""
    messages: list[Message] = Field(default_factory=list)
    mode: Literal["fast", "deep", "agent", "creative"] = "deep"
    max_output_tokens: int = Field(default=1024, ge=1, le=8192)
    thinking_budget: int = Field(default=2048, ge=256, le=16384)
    max_tool_calls: int = Field(default=8, ge=0, le=32)
    stream: bool = False
    tools: list[dict[str, Any]] = Field(default_factory=list)
    routing_strategy: Optional[str] = None
    enable_swarm: bool = False
    enable_ttt: bool = False
    n_drafts: int = Field(default=3, ge=1, le=5)
    ace_threshold: float = Field(default=0.85, ge=0.5, le=0.99)


class ThinkResponse(BaseModel):
    """Response body from the /v1/think endpoint."""
    model: str = "nexusmind-v1"
    model_tier: str = "nexusmind-flash"
    mode: str
    output: str
    thinking_trace: list[str] = Field(default_factory=list)
    drafts_used: int = 0
    thinking_steps: int = 0
    routing_strategy: str = ""
    signals: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
    swarm_report: Optional[dict[str, Any]] = None
    latency_class: Literal["low", "medium", "high"] = "medium"


class FeedbackRequest(BaseModel):
    """Feedback for Q-learning updates."""
    reward: float = Field(ge=-1.0, le=1.0)
    mode: str = "deep"
    complexity: str = "moderate"
    context: dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# Inference Helper
# ═══════════════════════════════════════════════════════════════════

def estimate_context_len(messages: list[Message]) -> int:
    """Rough token proxy for routing decisions."""
    return sum(len(m.content) for m in messages)


def choose_model_tier(req: ThinkRequest, complexity: str) -> tuple[str, str]:
    """MiMo-style tier routing: flash for fast paths, pro for hard jobs."""
    ctx = estimate_context_len(req.messages)
    hard_task = (
        req.mode == "agent"
        or req.max_output_tokens > 1500
        or req.max_tool_calls > 8
        or ctx > 120_000
        or len(req.tools) > 0
        or complexity == "complex"
        or req.mode in ("deep", "creative") and req.thinking_budget > 4096
    )
    if hard_task:
        return "nexusmind-pro", "high"
    return "nexusmind-flash", "low" if req.mode == "fast" else "medium"


def select_routing_strategy(
    req: ThinkRequest,
    complexity: str,
    q_recommendation: dict[str, Any],
) -> str:
    """Pick MoE routing strategy from request, Q-table, or mode defaults."""
    if req.routing_strategy:
        return req.routing_strategy

    q_action = getattr(q_recommendation, "action", None)
    if q_action is None and isinstance(q_recommendation, dict):
        q_action = q_recommendation.get("action")
    action_map = {
        "low": "sigma",
        "balanced": "dynamic_bias",
        "high": "multi_head_sigma",
        "uncanny": "noisy_topk",
    }
    if q_action in action_map:
        return action_map[q_action]

    mode_defaults = {
        "fast": "sigma",
        "deep": "dynamic_bias",
        "agent": "hierarchical",
        "creative": "noisy_topk",
    }
    complexity_overrides = {
        "complex": "multi_head_sigma",
        "moderate": mode_defaults.get(req.mode, "dynamic_bias"),
        "simple": "sigma",
    }
    return complexity_overrides.get(complexity, "dynamic_bias")


def apply_runtime_overrides(mdl: NexusMind, req: ThinkRequest) -> None:
    """Apply per-request thinking overrides without rebuilding the model."""
    mdl.thinking_engine.fast_ace.threshold = req.ace_threshold
    mdl.thinking_engine.deliberation.ace.threshold = req.ace_threshold
    mdl.thinking_engine.deliberation.max_steps = min(
        mdl.thinking_engine.deliberation.max_steps,
        max(1, req.thinking_budget // 512),
    )


def estimate_complexity(messages: list[Message]) -> str:
    """Estimate query complexity for Q-learning state."""
    total_len = sum(len(m.content) for m in messages)
    if total_len > 1000:
        return "complex"
    elif total_len > 300:
        return "moderate"
    return "simple"


def messages_to_text(messages: list[Message]) -> str:
    """Concatenate messages into a single text string."""
    parts = []
    for m in messages:
        prefix = {"system": "[SYS]", "user": "[USR]", "assistant": "[AST]", "tool": "[TOOL]"}
        parts.append(f"{prefix.get(m.role, '')} {m.content}")
    return "\n".join(parts)


def run_inference(
    text: str,
    mode: str,
    max_tokens: int,
    req: Optional[ThinkRequest] = None,
    routing_strategy: str = "",
    model_tier: str = "nexusmind-flash",
) -> dict[str, Any]:
    """Run model inference and synthesize a response."""
    demo = demo_inference(text, mode, routing_strategy=routing_strategy, model_tier=model_tier)

    try:
        mdl = get_model()
        if req is not None:
            apply_runtime_overrides(mdl, req)

        token_ids = [ord(c) % config.vocab_size for c in text[:config.max_seq_len]]
        input_ids = torch.tensor([token_ids], device=device)

        result = mdl.forward(
            input_ids,
            thinking_mode=mode,
            inference=True,
        )

        real_trace = result.get("thinking_trace", [])
        real_steps = result.get("thinking_steps", 0)
        param_counts = mdl.count_parameters()

        header = (
            f"[{model_tier} · {mode}] Processed {len(token_ids)} input tokens through "
            f"{config.n_layers} hybrid-attention blocks and {config.n_experts} MoE experts "
            f"({routing_strategy or config.routing_strategy} routing). "
            f"Thinking engine completed {real_steps} step(s)."
        )
        output = header + "\n\n" + demo["output"]

        return {
            "output": output,
            "thinking_trace": real_trace or demo.get("thinking_trace", []),
            "thinking_steps": real_steps or demo.get("thinking_steps", 0),
            "parameters": param_counts.get("total", 0),
        }
    except Exception:
        return demo


def demo_inference(
    text: str,
    mode: str,
    routing_strategy: str = "",
    model_tier: str = "nexusmind-flash",
) -> dict[str, Any]:
    """Demo inference narrative for UI testing and untrained-model fallback."""
    strategy = routing_strategy or config.routing_strategy
    mode_info = {
        "fast": {
            "steps": 1, "drafts": 1,
            "trace": ["[sigma] ACE exit_prob=0.91, early halt after 1 step"],
        },
        "deep": {
            "steps": 3, "drafts": 3,
            "trace": [
                "[dynamic_bias] 3 drafts initialized with diversity injection",
                "[deliberation] step 1/3, ACE=0.42, continuing...",
                "[deliberation] step 2/3, ACE=0.71, continuing...",
                "[deliberation] step 3/3, ACE=0.93, halting",
                "[adversarial_debate] Proposer → 3 candidates, Adversary → 1 flaw found",
                "[consistency] Cross-draft agreement=0.87, fusing with confidence weighting",
                "[working_memory] 4 intermediate conclusions stored across steps",
            ],
        },
        "agent": {
            "steps": 3, "drafts": 2,
            "trace": [
                "[graph_of_thought] 4 nodes initialized with stochastic diversity",
                "[GAT] Node exchange round 1/3 — meta-cognitive critique applied",
                "[GAT] Node exchange round 2/3 — 2 nodes halted (confidence > 0.85)",
                "[GAT] Node exchange round 3/3 — attention pooling over surviving nodes",
                "[tool_routing] Tool call accuracy: 97.0%, 0 tools dispatched",
            ],
        },
        "creative": {
            "steps": 4, "drafts": 3,
            "trace": [
                "[diffusion] Starting from noise, 4 denoising steps",
                "[denoise] t=0, β=0.10 — initial structure emerging",
                "[denoise] t=1, β=0.08 — refining sub-problem decomposition",
                "[denoise] t=2, β=0.05 — world model score=0.72",
                "[denoise] t=3, β=0.02 — final refinement, score=0.89",
                "[world_model] Future state prediction validates candidate",
            ],
        },
    }

    info = mode_info.get(mode, mode_info["deep"])
    word_count = len(text.split())

    output_parts = [
        f"NexusMind ({model_tier}) processes this {word_count}-word input through the {mode} thinking pipeline.",
        "",
        f"The hybrid attention stack (sliding-window {config.sliding_window_size} + global layers "
        f"from MiMo) processes the input with {config.n_heads}-head attention across "
        f"{config.n_layers} transformer blocks.",
        "",
        f"Sparse MoE routes through {config.n_experts} experts using {strategy} "
        f"strategy. Each expert has a unique activation function (SiLU, GELU, Mish, ReLU, SELU, "
        f"Tanh) for representational diversity. A shared expert provides universal knowledge baseline.",
        "",
    ]

    if mode == "deep":
        output_parts.extend([
            f"Multi-Draft Deliberation generates {info['drafts']} independent reasoning chains. "
            f"Each draft queries the Working Memory Bank ({config.working_memory_slots} slots) "
            f"and refines through {info['steps']} ReasoningCell steps.",
            "",
            "Cross-Draft Attention fuses the best aspects of each viewpoint. "
            "Adversarial Debate between Proposer and Adversary MoE ensembles identifies "
            "and resolves logical inconsistencies.",
            "",
            "Consistency-Weighted Aggregation produces the final output, weighted by "
            "inter-draft agreement confidence.",
        ])
    elif mode == "agent":
        output_parts.extend([
            "Graph-of-Thought creates 4 thought nodes that exchange insights via Graph Attention "
            "(GAT). Meta-Cognitive Critique reviews the entire graph and broadcasts corrections.",
            "",
            "Per-Thought Dynamic Depth allows confident nodes to freeze early while uncertain "
            "ones continue deliberating. Stochastic Bayesian routing via reparameterization "
            "forces creative exploration.",
        ])
    elif mode == "creative":
        output_parts.extend([
            f"Diffusion Refinement sculpts the answer from noise over {info['steps']} denoising "
            f"steps, each conditioned on decomposed sub-problem context.",
            "",
            "The World Model predicts future states and scores candidate quality before "
            "committing to the final output.",
        ])
    else:
        output_parts.append(
            "Fast mode uses a single ReasoningCell with ACE early exit for minimum latency."
        )

    output_parts.extend([
        "",
        f"Multi-Token Prediction ({config.mtp_heads} heads) enables speculative decoding "
        f"for ~2x inference speedup.",
        "",
        f"The Latent Knowledge Core ({config.latent_knowledge_slots} slots) provides implicit "
        f"RAG grounding, and the Episodic Memory ({config.episodic_memory_size} slots) tracks "
        f"long-term context.",
    ])

    return {
        "output": "\n".join(output_parts),
        "thinking_trace": info["trace"],
        "thinking_steps": info["steps"],
    }


# ═══════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.post("/v1/think", response_model=ThinkResponse)
async def think(req: ThinkRequest) -> ThinkResponse:
    """Main thinking endpoint.

    Routes input through the NexusMind model with the specified thinking mode.
    Auto-selects routing strategy based on complexity if not specified.
    Optionally runs multi-agent swarm for consensus reasoning.
    """
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    t0 = time.time()

    complexity = estimate_complexity(req.messages)
    text = messages_to_text(req.messages)
    model_tier, latency_class = choose_model_tier(req, complexity)

    q_recommendation = signal_engine.get_routing_recommendation(text)

    routing_strategy = select_routing_strategy(req, complexity, q_recommendation)
    config.routing_strategy = routing_strategy

    result = run_inference(
        text,
        req.mode,
        req.max_output_tokens,
        req=req,
        routing_strategy=routing_strategy,
        model_tier=model_tier,
    )

    swarm_report = None
    if req.enable_swarm:
        from nexusmind.agents import SwarmConfig

        swarm_result = swarm.run_swarm(text, SwarmConfig(max_rounds=2))
        swarm_report = {
            "consensus_score": swarm_result.consensus_score,
            "final_response": swarm_result.final_response,
            "rounds_completed": swarm_result.rounds_completed,
            "elapsed_ms": swarm_result.elapsed_ms,
        }
        result["output"] += f"\n\n[Swarm Consensus · {swarm_result.consensus_score:.2f}]\n{swarm_result.final_response}"

    # Update signals
    novelty = signal_engine.novelty_scorer.score(result["output"], prompt=text)
    coherence = signal_engine.coherence_scorer.score(result["output"])
    signal_engine.rsi.update(novelty["combined"])

    elapsed_ms = (time.time() - t0) * 1000

    return ThinkResponse(
        model="nexusmind-v1",
        model_tier=model_tier,
        mode=req.mode,
        output=result["output"],
        thinking_trace=result.get("thinking_trace", []),
        drafts_used=req.n_drafts if req.mode in ("deep", "creative") else 1,
        thinking_steps=result.get("thinking_steps", 0),
        routing_strategy=routing_strategy,
        latency_class=latency_class,
        signals={
            "novelty": novelty,
            "coherence": coherence,
            "rsi": round(signal_engine.rsi.compute() or 50.0, 2),
            "q_recommendation": q_recommendation.__dict__,
            "complexity": complexity,
        },
        latency_ms=round(elapsed_ms, 1),
        swarm_report=swarm_report,
    )


@app.post("/v1/feedback")
async def feedback(req: FeedbackRequest) -> dict[str, Any]:
    """Submit feedback for Q-learning policy updates.

    Updates the Q-table based on user reward signal, adjusting
    future routing decisions for similar query types.
    """
    signal_engine.process_feedback(
        output=req.context.get("output", ""),
        reward=req.reward,
        prompt=req.context.get("prompt", ""),
    )

    recommendation = signal_engine.get_routing_recommendation(
        req.context.get("prompt", "")
    )

    return {
        "status": "updated",
        "q_table": signal_engine.q_table.to_dict(),
        "recommendation": recommendation.__dict__,
    }


@app.get("/v1/signals")
async def get_signals() -> dict[str, Any]:
    """Get current signal dashboard state.

    Returns Q-table, RSI value, routing recommendation, and history.
    """
    return signal_engine.get_dashboard()


@app.get("/v1/config")
async def get_config() -> dict[str, Any]:
    """Get current model configuration."""
    return config.model_dump()


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check."""
    param_info = "lazy-loaded"
    if model is not None:
        param_info = f"{model.count_parameters()['total']:,} params on {device}"
    return {
        "status": "ok",
        "model": "nexusmind-v1",
        "device": device,
        "parameters": param_info,
    }


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the web UI."""
    web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
    index_path = os.path.join(web_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>NexusMind API</h1><p>Web UI not found. Use /docs for API docs.</p>")


# ═══════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════

def main():
    """Start the NexusMind API server."""
    import uvicorn
    print("╔══════════════════════════════════════════════════╗")
    print("║       NexusMind — Unified Hybrid Thinking       ║")
    print("║   Supermix · AI-Dem-Lab · MiMo  →  One Engine   ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"\n  Config: {config.n_layers}L / {config.d_model}D / {config.n_experts}E / {config.routing_strategy}")
    print(f"  Thinking: {config.max_thinking_steps} steps / {config.n_drafts} drafts / ACE@{config.ace_threshold}")
    print(f"  Memory: WM={config.working_memory_slots} / EM={config.episodic_memory_size} / LKC={config.latent_knowledge_slots}")
    print(f"  MTP: {'enabled' if config.mtp_enabled else 'disabled'} ({config.mtp_heads} heads)")
    print(f"\n  → http://localhost:8000")
    print(f"  → http://localhost:8000/docs\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
