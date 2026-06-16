"""Signal intelligence system for NexusMind.

Ported from the AI-Dem-Lab's interactive research playground into a
production-grade Python module.  Contains:

- QTable:          Q-learning policy for routing / temperature selection
- RSICalculator:   Relative Strength Index momentum on novelty streams
- EntropyRouter:   Dynamic entropy-source selection
- NoveltyScorer:   Measures output surprise via token diversity & n-grams
- CoherenceScorer: Measures output quality via repetition & structure
- SignalEngine:    Orchestrates all signal subsystems
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import secrets
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTIONS: Tuple[str, ...] = ("low", "balanced", "high", "uncanny")
"""Routing strategies that double as temperature presets."""

ACTION_PRESETS: Dict[str, Dict[str, float]] = {
    "low":      {"temperature": 0.15, "top_p": 0.65},
    "balanced": {"temperature": 0.80, "top_p": 0.90},
    "high":     {"temperature": 1.20, "top_p": 0.96},
    "uncanny":  {"temperature": 1.70, "top_p": 0.99},
}

ENTROPY_MODES = ("deterministic", "seeded_prng", "crypto", "mixed", "qrng_hook")

STOPWORDS: frozenset[str] = frozenset(
    "the a an and or to of in on for with this that is are be as by it from "
    "at we you i they not but if then can could should would about into over "
    "under when what why how yes no".split()
)

RSI_OVERBOUGHT: float = 70.0
RSI_OVERSOLD: float = 30.0


# ═══════════════════════════════════════════════════════════════════════════
#  QTable – Q-learning policy table
# ═══════════════════════════════════════════════════════════════════════════

class QTable:
    """Tabular Q-learning policy.

    State space:  ``(goal, entropy_mode, complexity_bucket)``
    Action space: routing strategies / temperature presets.
    """

    def __init__(
        self,
        actions: Sequence[str] = ACTIONS,
        default_value: float = 0.1,
    ) -> None:
        self.actions: Tuple[str, ...] = tuple(actions)
        self.default_value = default_value
        self._table: Dict[str, Dict[str, float]] = {}

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _key(state: Tuple[str, ...]) -> str:
        """Canonical string key for a state tuple."""
        return "::".join(str(s) for s in state)

    def _ensure(self, state: Tuple[str, ...]) -> Dict[str, float]:
        key = self._key(state)
        if key not in self._table:
            self._table[key] = {a: self.default_value for a in self.actions}
        return self._table[key]

    # -- core API ----------------------------------------------------------

    def choose_action(
        self,
        state: Tuple[str, ...],
        epsilon: float = 0.2,
    ) -> str:
        """Epsilon-greedy action selection."""
        if random.random() < epsilon:
            return random.choice(self.actions)
        return self.best_action(state)

    def best_action(self, state: Tuple[str, ...]) -> str:
        """Return the greedy (highest-Q) action for *state*."""
        q_row = self._ensure(state)
        return max(q_row, key=q_row.get)  # type: ignore[arg-type]

    def update(
        self,
        state: Tuple[str, ...],
        action: str,
        reward: float,
        next_state: Tuple[str, ...],
        alpha: float = 0.20,
        gamma: float = 0.85,
    ) -> float:
        """One-step Q-update rule.

        ``Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]``

        Returns the new Q-value.
        """
        q_row = self._ensure(state)
        next_row = self._ensure(next_state)
        old_q = q_row.get(action, self.default_value)
        max_next = max(next_row.values())
        new_q = old_q + alpha * (reward + gamma * max_next - old_q)
        q_row[action] = new_q
        return new_q

    def get_q(self, state: Tuple[str, ...], action: str) -> float:
        return self._ensure(state).get(action, self.default_value)

    def get_row(self, state: Tuple[str, ...]) -> Dict[str, float]:
        return dict(self._ensure(state))

    # -- serialization -----------------------------------------------------

    def to_json(self) -> str:
        """Serialize the Q-table to a JSON string."""
        payload = {
            "actions": list(self.actions),
            "default_value": self.default_value,
            "table": self._table,
        }
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json(cls, data: str) -> "QTable":
        """Deserialize a Q-table from a JSON string."""
        parsed = json.loads(data)
        qt = cls(
            actions=tuple(parsed["actions"]),
            default_value=parsed.get("default_value", 0.1),
        )
        qt._table = parsed.get("table", {})
        return qt

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "QTable":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())

    # -- inspection --------------------------------------------------------

    def __repr__(self) -> str:
        return f"QTable(states={len(self._table)}, actions={self.actions})"

    @property
    def states(self) -> List[str]:
        return list(self._table.keys())


# ═══════════════════════════════════════════════════════════════════════════
#  RSICalculator – Relative Strength Index momentum
# ═══════════════════════════════════════════════════════════════════════════

class RSIClassification(str, Enum):
    OVERBOUGHT = "overbought"
    OVERSOLD = "oversold"
    NEUTRAL = "neutral"


class RSICalculator:
    """Tracks a novelty-score stream and computes RSI (0-100).

    RSI is repurposed here as a momentum oscillator for output novelty:
    >70 = overbought (too volatile), <30 = oversold (too tame).
    """

    def __init__(self, max_history: int = 200) -> None:
        self._history: deque[float] = deque(maxlen=max_history)
        self.max_history = max_history

    def update(self, score: float) -> None:
        """Append a new novelty data-point."""
        self._history.append(float(score))

    def compute(self, window: int = 14) -> Optional[float]:
        """Return the RSI value (0-100), or ``None`` if insufficient data."""
        series = list(self._history)
        if len(series) < window + 1:
            return None

        deltas = [series[i] - series[i - 1] for i in range(1, len(series))]
        recent = deltas[-window:]

        gains = sum(d for d in recent if d >= 0)
        losses = sum(abs(d) for d in recent if d < 0)

        avg_gain = gains / window
        avg_loss = losses / window if losses > 0 else 1e-9

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def classify(self, window: int = 14) -> RSIClassification:
        """Return overbought / oversold / neutral classification."""
        rsi = self.compute(window)
        if rsi is None:
            return RSIClassification.NEUTRAL
        if rsi > RSI_OVERBOUGHT:
            return RSIClassification.OVERBOUGHT
        if rsi < RSI_OVERSOLD:
            return RSIClassification.OVERSOLD
        return RSIClassification.NEUTRAL

    @property
    def history(self) -> List[float]:
        return list(self._history)

    @property
    def current(self) -> Optional[float]:
        return self._history[-1] if self._history else None

    def __repr__(self) -> str:
        rsi = self.compute()
        rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
        return f"RSICalculator(points={len(self._history)}, rsi={rsi_str})"


# ═══════════════════════════════════════════════════════════════════════════
#  EntropyRouter – Dynamic entropy source selection
# ═══════════════════════════════════════════════════════════════════════════

class EntropyRouter:
    """Provides entropy values from configurable sources.

    Supported modes:
        * ``deterministic`` – always returns 0.5 (no randomness)
        * ``seeded_prng``   – reproducible Mulberry32-style PRNG
        * ``crypto``        – ``os.urandom`` / ``secrets``
        * ``mixed``         – 45% seeded PRNG + 55% crypto
        * ``qrng_hook``     – crypto stand-in (swap for real QRNG endpoint)
    """

    def __init__(self) -> None:
        self._prng_states: Dict[str, int] = {}

    @staticmethod
    def _mulberry32(seed: int) -> float:
        """Single step of the Mulberry32 PRNG, returns float in [0, 1)."""
        t = (seed + 0x6D2B79F5) & 0xFFFFFFFF
        r = ((t ^ (t >> 15)) * (1 | t)) & 0xFFFFFFFF
        r = (r + ((r ^ (r >> 7)) * (61 | r)) & 0xFFFFFFFF) & 0xFFFFFFFF
        r = (r ^ (r >> 14)) & 0xFFFFFFFF
        return r / 4294967296.0

    @staticmethod
    def _hash_seed(seed_str: str) -> int:
        """FNV-1a-inspired hash → unsigned 32-bit integer."""
        h = 2166136261
        for ch in seed_str:
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
        return h

    def _step_prng(self, seed_key: str) -> float:
        """Advance the seeded PRNG for *seed_key* by one step."""
        current = self._prng_states.get(seed_key)
        if current is None:
            current = self._hash_seed(seed_key)
        result = self._mulberry32(current)
        self._prng_states[seed_key] = (current + 0x6D2B79F5) & 0xFFFFFFFF
        return result

    @staticmethod
    def _crypto_rand() -> float:
        """Return a cryptographically random float in [0, 1)."""
        return int.from_bytes(os.urandom(4), "big") / 4294967296.0

    def get_entropy(
        self,
        mode: str = "crypto",
        seed: Optional[str] = None,
    ) -> float:
        """Return an entropy value in [0, 1) according to *mode*.

        Parameters
        ----------
        mode : str
            One of ``ENTROPY_MODES``.
        seed : str, optional
            Seed string for ``seeded_prng`` and ``mixed`` modes.
        """
        if mode == "deterministic":
            return 0.5

        seed_key = seed or "default"

        if mode == "seeded_prng":
            return self._step_prng(seed_key)

        if mode == "mixed":
            prng_part = self._step_prng(seed_key) * 0.45
            crypto_part = self._crypto_rand() * 0.55
            return prng_part + crypto_part

        if mode == "qrng_hook":
            # Placeholder: in production, replace with a real QRNG endpoint.
            return self._crypto_rand()

        # Default: crypto
        return self._crypto_rand()

    def reset(self, seed_key: Optional[str] = None) -> None:
        """Reset PRNG state for a seed (or all seeds)."""
        if seed_key:
            self._prng_states.pop(seed_key, None)
        else:
            self._prng_states.clear()


# ═══════════════════════════════════════════════════════════════════════════
#  NoveltyScorer – Measures output surprise
# ═══════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    import re
    return [t for t in re.sub(r"[^a-z0-9\s']", " ", text.lower()).split() if t]


class NoveltyScorer:
    """Quantifies how *surprising* a model output is.

    Sub-scores:
        1. **Token diversity ratio** – unique / total (excluding stopwords)
        2. **N-gram uniqueness** – fraction of unique bigrams + trigrams
        3. **Semantic distance from prompt** – 1 − Jaccard similarity
    """

    def __init__(self) -> None:
        self._history: List[float] = []

    def _token_diversity(self, tokens: List[str]) -> float:
        content = [t for t in tokens if t not in STOPWORDS]
        if not content:
            return 0.0
        return len(set(content)) / len(content)

    def _ngram_uniqueness(self, tokens: List[str], n_values: Sequence[int] = (2, 3)) -> float:
        if len(tokens) < 3:
            return 0.0
        scores: List[float] = []
        for n in n_values:
            ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
            if ngrams:
                scores.append(len(set(ngrams)) / len(ngrams))
        return sum(scores) / len(scores) if scores else 0.0

    def _semantic_distance(self, prompt_tokens: List[str], output_tokens: List[str]) -> float:
        """1 − Jaccard similarity between prompt and output token sets."""
        a = set(t for t in prompt_tokens if t not in STOPWORDS)
        b = set(t for t in output_tokens if t not in STOPWORDS)
        if not a and not b:
            return 0.0
        union = a | b
        inter = a & b
        jaccard = len(inter) / len(union) if union else 0.0
        return 1.0 - jaccard

    def score(
        self,
        output: str,
        prompt: str = "",
    ) -> Dict[str, float]:
        """Compute novelty sub-scores and a combined score (0-100).

        Returns dict with keys: ``diversity``, ``ngram_unique``,
        ``semantic_dist``, ``combined``.
        """
        out_tokens = _tokenize(output)
        prompt_tokens = _tokenize(prompt) if prompt else []

        diversity = self._token_diversity(out_tokens)
        ngram_unique = self._ngram_uniqueness(out_tokens)
        semantic_dist = self._semantic_distance(prompt_tokens, out_tokens)

        # Weighted combination (matching AI-Dem-Lab proportions)
        combined = max(0.0, min(100.0, (
            diversity * 0.40 + ngram_unique * 0.35 + semantic_dist * 0.25
        ) * 100.0))

        self._history.append(combined)
        return {
            "diversity": round(diversity, 4),
            "ngram_unique": round(ngram_unique, 4),
            "semantic_dist": round(semantic_dist, 4),
            "combined": round(combined, 2),
        }

    @property
    def history(self) -> List[float]:
        return list(self._history)


# ═══════════════════════════════════════════════════════════════════════════
#  CoherenceScorer – Measures output quality
# ═══════════════════════════════════════════════════════════════════════════

class CoherenceScorer:
    """Quantifies how *coherent / high-quality* a model output is.

    Sub-scores:
        1. **Repetition penalty** – penalizes repeated tokens
        2. **Structural scoring** – rewards sentence-ending punctuation
        3. **Self-consistency** – penalizes variance in sentence lengths
    """

    def __init__(self) -> None:
        self._history: List[float] = []

    def _repetition_penalty(self, tokens: List[str]) -> float:
        """Returns penalty in [0, 1]; 0 = no repeats, 1 = all repeats."""
        if not tokens:
            return 0.0
        repeats = len(tokens) - len(set(tokens))
        return min(1.0, repeats / max(1, len(tokens)))

    def _structural_score(self, text: str) -> float:
        """Reward proper sentence structure. Returns [0, 1]."""
        import re
        sentences = re.split(r"[.!?]+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        # Reward: has punctuation, multiple sentences, reasonable length
        punct_count = len(re.findall(r"[.!?]", text))
        word_count = len(text.split())
        punct_ratio = min(1.0, punct_count / max(1, word_count / 15))
        multi_sentence = min(1.0, len(sentences) / 5.0)
        return (punct_ratio * 0.6 + multi_sentence * 0.4)

    def _self_consistency(self, text: str) -> float:
        """Lower variance in sentence lengths = higher consistency.

        Returns [0, 1]; 1 = perfectly uniform.
        """
        import re
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if len(sentences) < 2:
            return 0.5
        lengths = [len(s.split()) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        std = math.sqrt(variance)
        # Normalize: std < 3 → excellent, std > 15 → poor
        return max(0.0, min(1.0, 1.0 - (std / 15.0)))

    def score(self, output: str) -> Dict[str, float]:
        """Compute coherence sub-scores and combined score (0-100).

        Returns dict with keys: ``repetition``, ``structure``,
        ``consistency``, ``combined``.
        """
        tokens = _tokenize(output)

        rep_penalty = self._repetition_penalty(tokens)
        structure = self._structural_score(output)
        consistency = self._self_consistency(output)

        combined = max(0.0, min(100.0, (
            (1.0 - rep_penalty) * 0.40 + structure * 0.35 + consistency * 0.25
        ) * 100.0))

        self._history.append(combined)
        return {
            "repetition": round(rep_penalty, 4),
            "structure": round(structure, 4),
            "consistency": round(consistency, 4),
            "combined": round(combined, 2),
        }

    @property
    def history(self) -> List[float]:
        return list(self._history)


# ═══════════════════════════════════════════════════════════════════════════
#  SignalEngine – Orchestrates all signal subsystems
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RoutingRecommendation:
    """Structured recommendation from the signal engine."""
    action: str
    temperature: float
    top_p: float
    entropy_mode: str
    rsi_value: Optional[float]
    rsi_class: str
    novelty: float
    coherence: float
    q_values: Dict[str, float]
    rationale: str


class SignalEngine:
    """Unified signal orchestrator.

    Combines Q-learning, RSI momentum, entropy routing, novelty, and
    coherence scoring into a single decision-making surface.
    """

    def __init__(
        self,
        alpha: float = 0.20,
        gamma: float = 0.85,
        epsilon: float = 0.20,
        rsi_window: int = 14,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rsi_window = rsi_window

        self.q_table = QTable()
        self.rsi = RSICalculator()
        self.entropy_router = EntropyRouter()
        self.novelty_scorer = NoveltyScorer()
        self.coherence_scorer = CoherenceScorer()

        self._feedback_count: int = 0
        self._history: List[Dict[str, Any]] = []

    # -- complexity bucketing -----------------------------------------------

    @staticmethod
    def _complexity_bucket(text: str) -> str:
        """Bucket prompt complexity into low/medium/high."""
        words = text.split()
        n = len(words)
        if n < 15:
            return "low"
        elif n < 60:
            return "medium"
        return "high"

    # -- core API ----------------------------------------------------------

    def process_feedback(
        self,
        output: str,
        reward: float,
        prompt: str = "",
        goal: str = "scientific",
        entropy_mode: str = "crypto",
    ) -> Dict[str, Any]:
        """Update all signal subsystems from user / auto feedback.

        Parameters
        ----------
        output : str
            The model output text.
        reward : float
            Reward signal (positive = good, negative = bad).
        prompt : str
            Original prompt (used for novelty scoring).
        goal : str
            Current goal/style setting.
        entropy_mode : str
            Current entropy source mode.

        Returns
        -------
        dict with keys: novelty, coherence, rsi, q_update, state.
        """
        # Score output
        novelty = self.novelty_scorer.score(output, prompt)
        coherence = self.coherence_scorer.score(output)

        # Feed RSI
        self.rsi.update(novelty["combined"])

        # Build state & update Q
        complexity = self._complexity_bucket(prompt)
        state = (goal, entropy_mode, complexity)
        next_state = state  # single-step: next = current

        # Infer the action from Q-table (we assume last recommended action)
        action = self.q_table.best_action(state)
        new_q = self.q_table.update(
            state, action, reward, next_state,
            alpha=self.alpha, gamma=self.gamma,
        )

        self._feedback_count += 1

        record = {
            "timestamp": None,  # caller can fill
            "reward": reward,
            "novelty": novelty["combined"],
            "coherence": coherence["combined"],
            "action": action,
            "goal": goal,
            "entropy_mode": entropy_mode,
            "rsi": self.rsi.compute(self.rsi_window),
        }
        self._history.append(record)
        if len(self._history) > 200:
            self._history = self._history[-200:]

        return {
            "novelty": novelty,
            "coherence": coherence,
            "rsi": self.rsi.compute(self.rsi_window),
            "rsi_class": self.rsi.classify(self.rsi_window).value,
            "q_update": new_q,
            "state": "::".join(state),
        }

    def get_routing_recommendation(
        self,
        prompt: str = "",
        goal: str = "scientific",
        entropy_mode: str = "crypto",
    ) -> RoutingRecommendation:
        """Return the best routing strategy given current context.

        Uses Q-table for action selection, RSI for momentum awareness,
        and entropy routing for randomness injection.
        """
        complexity = self._complexity_bucket(prompt)
        state = (goal, entropy_mode, complexity)

        # RSI-aware epsilon adjustment: if overbought, reduce exploration
        rsi_val = self.rsi.compute(self.rsi_window)
        rsi_cls = self.rsi.classify(self.rsi_window)

        adjusted_epsilon = self.epsilon
        if rsi_cls == RSIClassification.OVERBOUGHT:
            adjusted_epsilon = max(0.02, self.epsilon * 0.3)
        elif rsi_cls == RSIClassification.OVERSOLD:
            adjusted_epsilon = min(0.6, self.epsilon * 1.5)

        action = self.q_table.choose_action(state, epsilon=adjusted_epsilon)
        preset = ACTION_PRESETS.get(action, ACTION_PRESETS["balanced"])

        q_values = self.q_table.get_row(state)
        nov_hist = self.novelty_scorer.history
        coh_hist = self.coherence_scorer.history

        rationale_parts = [f"Q-policy selects '{action}' for {goal}/{entropy_mode}/{complexity}."]
        if rsi_val is not None:
            rationale_parts.append(f"RSI={rsi_val:.1f} ({rsi_cls.value}).")
        if rsi_cls == RSIClassification.OVERBOUGHT:
            rationale_parts.append("Epsilon reduced to stabilize.")
        elif rsi_cls == RSIClassification.OVERSOLD:
            rationale_parts.append("Epsilon increased to explore.")

        return RoutingRecommendation(
            action=action,
            temperature=preset["temperature"],
            top_p=preset["top_p"],
            entropy_mode=entropy_mode,
            rsi_value=rsi_val,
            rsi_class=rsi_cls.value,
            novelty=nov_hist[-1] if nov_hist else 0.0,
            coherence=coh_hist[-1] if coh_hist else 0.0,
            q_values=q_values,
            rationale=" ".join(rationale_parts),
        )

    def get_dashboard(self) -> Dict[str, Any]:
        """Return a snapshot of all signal states for monitoring."""
        rsi_val = self.rsi.compute(self.rsi_window)
        return {
            "q_table": {
                "states": self.q_table.states,
                "num_states": len(self.q_table.states),
            },
            "rsi": {
                "value": rsi_val,
                "classification": self.rsi.classify(self.rsi_window).value,
                "history_length": len(self.rsi.history),
            },
            "novelty": {
                "latest": self.novelty_scorer.history[-1] if self.novelty_scorer.history else None,
                "history_length": len(self.novelty_scorer.history),
            },
            "coherence": {
                "latest": self.coherence_scorer.history[-1] if self.coherence_scorer.history else None,
                "history_length": len(self.coherence_scorer.history),
            },
            "feedback_count": self._feedback_count,
            "config": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "rsi_window": self.rsi_window,
            },
        }

    def save_state(self, path: str) -> None:
        """Persist the full signal engine state to a JSON file."""
        payload = {
            "q_table": json.loads(self.q_table.to_json()),
            "rsi_history": self.rsi.history,
            "novelty_history": self.novelty_scorer.history,
            "coherence_history": self.coherence_scorer.history,
            "feedback_count": self._feedback_count,
            "history": self._history[-100:],
            "config": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "rsi_window": self.rsi_window,
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_state(cls, path: str) -> "SignalEngine":
        """Restore a signal engine from a saved JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cfg = data.get("config", {})
        engine = cls(
            alpha=cfg.get("alpha", 0.20),
            gamma=cfg.get("gamma", 0.85),
            epsilon=cfg.get("epsilon", 0.20),
            rsi_window=cfg.get("rsi_window", 14),
        )
        engine.q_table = QTable.from_json(json.dumps(data["q_table"]))
        for v in data.get("rsi_history", []):
            engine.rsi.update(v)
        engine.novelty_scorer._history = data.get("novelty_history", [])
        engine.coherence_scorer._history = data.get("coherence_history", [])
        engine._feedback_count = data.get("feedback_count", 0)
        engine._history = data.get("history", [])
        return engine

    def __repr__(self) -> str:
        return (
            f"SignalEngine(feedback={self._feedback_count}, "
            f"q_states={len(self.q_table.states)}, "
            f"rsi={self.rsi.compute(self.rsi_window)})"
        )
