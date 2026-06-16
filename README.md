# NexusMind — Unified Hybrid Thinking Engine

> **26 generations of neural architecture innovations fused into one model.**

NexusMind is a unified hybrid thinking API that fuses innovations from three projects:

| Source | Innovations |
|--------|-------------|
| **[Supermix](https://github.com/kai9987kai/Supermix)** (v8–v26) | 6 MoE routing strategies, Recursive Thought + ACE, Multi-Draft Deliberation, Graph-of-Thought, Adversarial Debate, Diffusion Refinement, Spiking Neurons, TTT |
| **[AI-Dem-Lab](https://github.com/kai9987kai/AI-Dem-Lab)** | Q-Learning policy, RSI momentum, entropy routing, multi-agent swarm |
| **[Xiaomi MiMo](https://mimo.xiaomi.com/)** | Hybrid attention (sliding-window + global), Multi-Token Prediction, sparse MoE, dynamic bias routing |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NexusMind v1.0                              │
├────────────────┬────────────────┬────────────────┬─────────────────┤
│   Embeddings   │  N × Blocks    │  Think Engine  │  Output Heads   │
│  Token + Pos   │  HybridAttn    │  Fast / Deep   │  LM Head        │
│                │  + MoE FFN     │  Agent / Creat │  MTP Heads      │
├────────────────┴────────────────┴────────────────┴─────────────────┤
│                        Memory Systems                               │
│  Working Memory (v19) │ Episodic (v22) │ Latent Knowledge Core (v20)│
├─────────────────────────────────────────────────────────────────────┤
│                    Signal Intelligence                              │
│  Q-Learning Policy │ RSI Oscillator │ Multi-Agent Swarm             │
└─────────────────────────────────────────────────────────────────────┘
```

## Thinking Modes

| Mode | Strategy | Best For |
|------|----------|----------|
| ⚡ **Fast** | Single ReasoningCell + ACE early exit | Low-latency queries |
| 🔬 **Deep** | Multi-Draft Deliberation + Adversarial Debate | Complex reasoning |
| 🤖 **Agent** | Graph-of-Thought + tool routing | Multi-step tasks |
| 🎨 **Creative** | Diffusion Refinement + World Model | Open-ended generation |

## MoE Routing Strategies

| Strategy | Source | Key Innovation |
|----------|--------|----------------|
| Noisy Top-K | Supermix v8 | Learnable noise for exploration |
| Hierarchical | Supermix v9 / DeepSeek-MoE | Two-level domain→expert routing |
| Dynamic Bias | Supermix v10 / DeepSeek-V3 / MiMo | Aux-loss-free load balancing |
| Expert Choice | Supermix v11 | Experts pick tokens (reversed routing) |
| Sigma | Supermix v12 | Independent sigmoid scores |
| Multi-Head Sigma | Supermix v14 | Routing committee with multiple heads |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
python -m nexusmind.api

# Or import directly
python -c "from nexusmind import NexusMind, NexusConfig; m = NexusMind(NexusConfig()); print(m.count_parameters())"
```

## API Endpoints

```
POST /v1/think     → Main thinking endpoint
POST /v1/feedback  → Q-learning feedback
GET  /v1/signals   → Signal dashboard
GET  /v1/config    → Model configuration
GET  /health       → Health check
GET  /             → Web UI
```

### Example Request

```bash
curl -X POST http://localhost:8000/v1/think \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain MoE routing"}],
    "mode": "deep",
    "thinking_budget": 2048,
    "n_drafts": 3
  }'
```

## Project Structure

```
NexusMind/
├── nexusmind/
│   ├── __init__.py      # Package entry
│   ├── config.py        # All hyperparameters (Pydantic)
│   ├── core.py          # Main NexusMind model
│   ├── attention.py     # Hybrid sliding-window + global attention
│   ├── routing.py       # 6 MoE routing strategies
│   ├── thinking.py      # Thinking engine (deliberation, debate, diffusion)
│   ├── memory.py        # Working + Episodic + Latent Knowledge memory
│   ├── mtp.py           # Multi-Token Prediction heads
│   ├── signals.py       # Q-learning, RSI, entropy, novelty, coherence
│   ├── agents.py        # Multi-agent swarm orchestration
│   └── api.py           # FastAPI server
├── web/
│   └── index.html       # Premium glassmorphism web UI
├── requirements.txt
└── README.md
```

## Innovation Summary

### From Supermix Expert Heads (26 generations)
- **v8**: Noisy Top-K Gating MoE
- **v9**: Hierarchical Two-Level Routing + Shared Expert
- **v10**: Aux-Loss-Free Dynamic Bias Load Balancing
- **v11**: Expert Choice Routing (experts pick tokens)
- **v12**: Sigma Gating (independent sigmoid scores)
- **v13**: Iterative Reasoning + Cross-Expert Attention
- **v14**: Recursive Thought + ACE + Multi-Head Sigma
- **v15**: Reflexive Two-Pass Self-Correction
- **v16**: MetaCognitive Iterative Reflection
- **v17**: Latent Beam Search (Tree-of-Thought)
- **v18**: Consensus via Architectural Diversity
- **v19**: Multi-Draft Deliberation + Working Memory
- **v20**: Graph-of-Thought + Latent Knowledge Core
- **v21**: Adversarial Self-Play + Hierarchical Abstraction
- **v22**: Hypernetwork Experts + Episodic Memory
- **v23**: Diffusion Refinement + World Model
- **v24**: Neural ODE + Quantum Superposition
- **v25**: Fractal Recursion + Hyperbolic Geometry
- **v26**: Spiking Neurons + Liquid Synapses + TTT

### From AI-Dem-Lab
- Q-Learning adaptive policy routing
- RSI momentum oscillator for output stability
- Multi-source entropy routing
- 5-agent swarm (Generator, Critic, Skeptic, Archivist, Anomaly Hunter)

### From Xiaomi MiMo
- Hybrid Attention (sliding-window + global interleaved)
- Multi-Token Prediction for speculative decoding
- Sparse MoE with shared expert pattern
- Dynamic bias load balancing
- Agent-focused post-training

## License

MIT License — Copyright (c) 2026 kai9987kai
