"""Multi-agent swarm system for NexusMind.

Inspired by the AI-Dem-Lab's multi-perspective research framework and
Supermix's omni-collective prompt-variant architecture.  Each specialised
agent applies a distinct cognitive lens, and the :class:`SwarmOrchestrator`
fuses their outputs into a consensus response.

Agent roles
-----------
* **GeneratorAgent** – produces candidate responses
* **CriticAgent** – evaluates quality, finds flaws
* **SkepticAgent** – questions assumptions, checks evidence
* **ArchivistAgent** – retrieves relevant context from memory
* **AnomalyHunterAgent** – detects unusual patterns & edge cases
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

AgentRole = Literal["generator", "critic", "skeptic", "archivist", "anomaly_hunter"]

AGENT_ROLES: Tuple[str, ...] = (
    "generator",
    "critic",
    "skeptic",
    "archivist",
    "anomaly_hunter",
)


@dataclass
class AgentResponse:
    """Structured output from a single agent."""
    role: str
    content: str
    confidence: float = 0.5
    flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0


@dataclass
class SharedState:
    """Mutable state shared across agents during a swarm round."""
    prompt: str = ""
    responses: List[AgentResponse] = field(default_factory=list)
    memory_context: List[str] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)
    round_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SwarmConfig:
    """Configuration for a swarm run."""
    max_rounds: int = 3
    execution_mode: Literal["round_robin", "parallel"] = "round_robin"
    consensus_threshold: float = 0.6
    enable_conflict_resolution: bool = True
    agent_roles: Tuple[str, ...] = AGENT_ROLES
    verbose: bool = False


@dataclass
class SwarmResult:
    """Final output from a swarm run."""
    final_response: str
    consensus_score: float
    rounds_completed: int
    agent_responses: List[AgentResponse]
    conflicts: List[Dict[str, Any]]
    history: List[Dict[str, Any]]
    elapsed_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Base Agent
# ═══════════════════════════════════════════════════════════════════════════

class Agent(ABC):
    """Abstract base class for all swarm agents.

    Each agent has a *role* string, internal state, and a ``process``
    method that takes an input context plus shared state and returns a
    structured :class:`AgentResponse`.
    """

    def __init__(self, role: str) -> None:
        self.role: str = role
        self._call_count: int = 0
        self._internal_state: Dict[str, Any] = {}

    @abstractmethod
    def process(
        self,
        input_context: str,
        shared_state: SharedState,
    ) -> AgentResponse:
        """Process the input and return a response."""
        ...

    def reset(self) -> None:
        """Reset internal state."""
        self._call_count = 0
        self._internal_state.clear()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(role={self.role!r}, calls={self._call_count})"


# ═══════════════════════════════════════════════════════════════════════════
#  Specialised Agents
# ═══════════════════════════════════════════════════════════════════════════

class GeneratorAgent(Agent):
    """Produces candidate responses.

    Strategy:
    * Extracts key terms from the prompt
    * Builds multiple draft variants using different framings
    * Returns the richest, most grounded draft
    """

    def __init__(self) -> None:
        super().__init__(role="generator")

    def process(
        self,
        input_context: str,
        shared_state: SharedState,
    ) -> AgentResponse:
        t0 = time.perf_counter()
        self._call_count += 1

        # Extract key terms for grounding
        tokens = _extract_content_tokens(input_context)
        key_terms = tokens[:8]

        # Build multi-variant drafts
        drafts: List[str] = []

        # Draft 1: Direct response
        drafts.append(
            f"Based on the request about {', '.join(key_terms[:3])}: "
            f"The core concepts involve {', '.join(key_terms)}. "
            f"A grounded response should address each element while "
            f"maintaining factual accuracy and structural clarity."
        )

        # Draft 2: Expanded with context
        memory_ctx = " ".join(shared_state.memory_context[:3]) if shared_state.memory_context else ""
        if memory_ctx:
            drafts.append(
                f"Drawing on available context ({memory_ctx[:100]}...): "
                f"The request centers on {', '.join(key_terms[:4])}. "
                f"Integrating prior knowledge strengthens the response."
            )

        # Select best draft (longest, most complete)
        best = max(drafts, key=len)

        # Track generation state
        self._internal_state["last_key_terms"] = key_terms
        self._internal_state["num_drafts"] = len(drafts)

        elapsed = (time.perf_counter() - t0) * 1000
        return AgentResponse(
            role=self.role,
            content=best,
            confidence=0.7,
            flags=["draft", f"variants:{len(drafts)}"],
            metadata={"key_terms": key_terms, "draft_count": len(drafts)},
            elapsed_ms=elapsed,
        )


class CriticAgent(Agent):
    """Evaluates quality and finds flaws in candidate responses.

    Checks for:
    * Unsupported claims (no grounding tokens)
    * Repetition
    * Missing coverage of prompt terms
    * Structural issues
    """

    def __init__(self) -> None:
        super().__init__(role="critic")

    def process(
        self,
        input_context: str,
        shared_state: SharedState,
    ) -> AgentResponse:
        t0 = time.perf_counter()
        self._call_count += 1

        # Gather all prior responses to critique
        prior = [r for r in shared_state.responses if r.role == "generator"]
        if not prior:
            elapsed = (time.perf_counter() - t0) * 1000
            return AgentResponse(
                role=self.role,
                content="No generator output to critique yet.",
                confidence=0.3,
                elapsed_ms=elapsed,
            )

        target = prior[-1]
        issues: List[str] = []
        strengths: List[str] = []

        # Check prompt coverage
        prompt_tokens = set(_extract_content_tokens(shared_state.prompt))
        response_tokens = set(_extract_content_tokens(target.content))
        coverage = len(prompt_tokens & response_tokens) / max(1, len(prompt_tokens))
        if coverage < 0.3:
            issues.append(f"Low prompt coverage ({coverage:.0%}): key terms may be missing.")
        else:
            strengths.append(f"Good prompt coverage ({coverage:.0%}).")

        # Check repetition
        words = target.content.lower().split()
        word_counts = Counter(words)
        repeated = [w for w, c in word_counts.items() if c > 3 and w not in _STOPWORDS]
        if repeated:
            issues.append(f"Excessive repetition of: {', '.join(repeated[:5])}.")

        # Check structural quality
        sentences = [s.strip() for s in re.split(r"[.!?]+", target.content) if s.strip()]
        if len(sentences) < 2:
            issues.append("Response is too short or lacks sentence structure.")
        else:
            strengths.append(f"Response has {len(sentences)} sentences.")

        # Build critique
        critique_parts: List[str] = []
        if strengths:
            critique_parts.append("Strengths: " + "; ".join(strengths) + ".")
        if issues:
            critique_parts.append("Issues: " + "; ".join(issues) + ".")
        else:
            critique_parts.append("No major issues found.")

        confidence = max(0.3, min(0.95, 1.0 - len(issues) * 0.15))

        elapsed = (time.perf_counter() - t0) * 1000
        return AgentResponse(
            role=self.role,
            content=" ".join(critique_parts),
            confidence=confidence,
            flags=[f"issues:{len(issues)}", f"coverage:{coverage:.2f}"],
            metadata={"issues": issues, "strengths": strengths, "coverage": coverage},
            elapsed_ms=elapsed,
        )


class SkepticAgent(Agent):
    """Questions assumptions and checks evidence.

    Applies a skeptical lens:
    * Flags unsupported superlatives
    * Identifies claims that need evidence
    * Checks for hedging / uncertainty markers
    """

    SUPERLATIVE_PATTERNS = re.compile(
        r"\b(always|never|certainly|definitely|proven|guaranteed|absolutely|"
        r"undoubtedly|100%|impossible|perfect|flawless)\b",
        re.IGNORECASE,
    )
    HEDGE_PATTERNS = re.compile(
        r"\b(may|might|could|possibly|potentially|suggests|appears|"
        r"likely|unlikely|uncertain|debatable)\b",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        super().__init__(role="skeptic")

    def process(
        self,
        input_context: str,
        shared_state: SharedState,
    ) -> AgentResponse:
        t0 = time.perf_counter()
        self._call_count += 1

        # Analyze all prior responses
        all_content = " ".join(r.content for r in shared_state.responses)
        if not all_content.strip():
            all_content = input_context

        # Find superlatives (overclaiming)
        superlatives = self.SUPERLATIVE_PATTERNS.findall(all_content)

        # Find hedging (good practice)
        hedges = self.HEDGE_PATTERNS.findall(all_content)

        # Build skeptical assessment
        concerns: List[str] = []
        observations: List[str] = []

        if superlatives:
            concerns.append(
                f"Found {len(superlatives)} absolute claim(s) "
                f"({', '.join(set(s.lower() for s in superlatives[:5]))}). "
                f"These may need evidence or softening."
            )

        if hedges:
            observations.append(
                f"Found {len(hedges)} hedge marker(s) – good epistemic practice."
            )
        else:
            concerns.append("No hedging or uncertainty markers found. Consider adding nuance.")

        # Check for citation-like patterns
        has_citations = bool(re.search(r"\[\d+\]|\(.*\d{4}\)", all_content))
        if has_citations:
            observations.append("Contains citation-style references.")
        else:
            concerns.append("No citations or references detected.")

        parts: List[str] = []
        if observations:
            parts.append("Observations: " + " ".join(observations))
        if concerns:
            parts.append("Skeptical concerns: " + " ".join(concerns))

        confidence = max(0.4, min(0.9, 0.8 - len(concerns) * 0.1))

        elapsed = (time.perf_counter() - t0) * 1000
        return AgentResponse(
            role=self.role,
            content=" ".join(parts) if parts else "No significant skeptical concerns.",
            confidence=confidence,
            flags=[f"superlatives:{len(superlatives)}", f"hedges:{len(hedges)}"],
            metadata={
                "superlatives": list(set(s.lower() for s in superlatives)),
                "hedges": list(set(h.lower() for h in hedges)),
                "has_citations": has_citations,
            },
            elapsed_ms=elapsed,
        )


class ArchivistAgent(Agent):
    """Retrieves relevant context from memory.

    In the absence of a real vector store, the archivist:
    * Extracts key terms from the prompt
    * Matches them against available shared-state memory
    * Surfaces the most relevant context passages
    * Tracks retrieval history for deduplication
    """

    def __init__(self) -> None:
        super().__init__(role="archivist")
        self._retrieval_log: List[str] = []

    def process(
        self,
        input_context: str,
        shared_state: SharedState,
    ) -> AgentResponse:
        t0 = time.perf_counter()
        self._call_count += 1

        prompt_tokens = set(_extract_content_tokens(input_context))
        memory = shared_state.memory_context

        # Score each memory entry by token overlap
        scored: List[Tuple[float, str]] = []
        for entry in memory:
            entry_tokens = set(_extract_content_tokens(entry))
            overlap = len(prompt_tokens & entry_tokens) / max(1, len(prompt_tokens | entry_tokens))
            scored.append((overlap, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        relevant = [entry for score, entry in scored if score > 0.05][:5]

        if relevant:
            content = (
                f"Retrieved {len(relevant)} relevant context entries. "
                f"Top match overlap: {scored[0][0]:.0%}. "
                f"Context: " + " | ".join(r[:100] for r in relevant)
            )
            confidence = min(0.9, 0.4 + scored[0][0])
        else:
            content = "No relevant memory context found for this prompt."
            confidence = 0.2

        self._retrieval_log.extend(relevant)

        elapsed = (time.perf_counter() - t0) * 1000
        return AgentResponse(
            role=self.role,
            content=content,
            confidence=confidence,
            flags=[f"retrieved:{len(relevant)}", f"memory_size:{len(memory)}"],
            metadata={"retrieved_count": len(relevant), "memory_size": len(memory)},
            elapsed_ms=elapsed,
        )

    def reset(self) -> None:
        super().reset()
        self._retrieval_log.clear()


class AnomalyHunterAgent(Agent):
    """Detects unusual patterns and edge cases.

    Looks for:
    * Token frequency anomalies
    * Unusual character distributions
    * Self-referential loops
    * Extreme length deviations
    * Contradiction patterns between agent responses
    """

    def __init__(self) -> None:
        super().__init__(role="anomaly_hunter")

    def process(
        self,
        input_context: str,
        shared_state: SharedState,
    ) -> AgentResponse:
        t0 = time.perf_counter()
        self._call_count += 1

        all_content = " ".join(r.content for r in shared_state.responses)
        if not all_content.strip():
            all_content = input_context

        anomalies: List[str] = []

        # 1. Token frequency anomaly
        tokens = all_content.lower().split()
        if tokens:
            counts = Counter(tokens)
            most_common = counts.most_common(1)[0]
            if most_common[1] > len(tokens) * 0.15 and most_common[0] not in _STOPWORDS:
                anomalies.append(
                    f"Token '{most_common[0]}' appears {most_common[1]} times "
                    f"({most_common[1]/len(tokens):.0%} of output) – possible repetition loop."
                )

        # 2. Unusual character distribution
        if all_content:
            special_ratio = sum(1 for c in all_content if not c.isalnum() and c != ' ') / max(1, len(all_content))
            if special_ratio > 0.15:
                anomalies.append(f"High special character ratio ({special_ratio:.0%}) – possible encoding issue.")

        # 3. Self-referential patterns
        self_ref = re.findall(r"\b(I am an? (?:AI|model|language model|assistant))\b", all_content, re.IGNORECASE)
        if len(self_ref) > 2:
            anomalies.append(f"Excessive self-referential statements ({len(self_ref)}x).")

        # 4. Extreme length
        if len(tokens) > 500:
            anomalies.append(f"Very long output ({len(tokens)} tokens) – may benefit from summarisation.")
        elif len(tokens) < 5 and shared_state.round_number > 0:
            anomalies.append(f"Very short output ({len(tokens)} tokens) – may be incomplete.")

        # 5. Confidence variance across agents
        confidences = [r.confidence for r in shared_state.responses if r.confidence > 0]
        if len(confidences) >= 2:
            conf_var = _variance(confidences)
            if conf_var > 0.04:
                anomalies.append(
                    f"High confidence variance ({conf_var:.3f}) across agents – "
                    f"possible disagreement."
                )

        # Store in shared state
        shared_state.anomalies.extend(anomalies)

        if anomalies:
            content = f"Detected {len(anomalies)} anomaly(ies): " + " ".join(anomalies)
        else:
            content = "No anomalies detected in current outputs."

        confidence = max(0.3, 0.9 - len(anomalies) * 0.1)

        elapsed = (time.perf_counter() - t0) * 1000
        return AgentResponse(
            role=self.role,
            content=content,
            confidence=confidence,
            flags=[f"anomalies:{len(anomalies)}"],
            metadata={"anomalies": anomalies},
            elapsed_ms=elapsed,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  SwarmOrchestrator
# ═══════════════════════════════════════════════════════════════════════════

AGENT_REGISTRY: Dict[str, type] = {
    "generator": GeneratorAgent,
    "critic": CriticAgent,
    "skeptic": SkepticAgent,
    "archivist": ArchivistAgent,
    "anomaly_hunter": AnomalyHunterAgent,
}


class SwarmOrchestrator:
    """Manages the multi-agent swarm workflow.

    Execution modes
    ---------------
    * **round_robin** – agents run sequentially, each building on
      the previous responses.
    * **parallel** – all agents run on the same initial context,
      then results are merged.

    Consensus
    ---------
    Uses confidence-weighted voting.  If the generator's output
    passes critic + skeptic checks with aggregate confidence
    above the threshold, it becomes the final answer.  Otherwise,
    the generator re-drafts in the next round.
    """

    def __init__(self, config: Optional[SwarmConfig] = None) -> None:
        self.config = config or SwarmConfig()
        self._agents: Dict[str, Agent] = {}
        self._history: List[Dict[str, Any]] = []
        self._build_agents()

    def _build_agents(self) -> None:
        """Instantiate agents based on config roles."""
        self._agents.clear()
        for role in self.config.agent_roles:
            cls = AGENT_REGISTRY.get(role)
            if cls is not None:
                self._agents[role] = cls()

    def run_swarm(
        self,
        prompt: str,
        config: Optional[SwarmConfig] = None,
        memory_context: Optional[List[str]] = None,
    ) -> SwarmResult:
        """Execute the full multi-agent workflow.

        Parameters
        ----------
        prompt : str
            The user prompt to process.
        config : SwarmConfig, optional
            Override config for this run.
        memory_context : list[str], optional
            External memory entries for the archivist.

        Returns
        -------
        SwarmResult
            The consensus response plus all agent outputs and metadata.
        """
        cfg = config or self.config
        t0 = time.perf_counter()

        shared = SharedState(
            prompt=prompt,
            memory_context=memory_context or [],
        )

        all_responses: List[AgentResponse] = []
        conflicts: List[Dict[str, Any]] = []
        round_history: List[Dict[str, Any]] = []

        for round_idx in range(cfg.max_rounds):
            shared.round_number = round_idx

            round_responses = self._execute_round(prompt, shared, cfg)
            all_responses.extend(round_responses)
            shared.responses.extend(round_responses)

            # Record round
            round_record = {
                "round": round_idx,
                "responses": [
                    {"role": r.role, "confidence": r.confidence, "flags": r.flags}
                    for r in round_responses
                ],
            }
            round_history.append(round_record)

            # Check consensus
            consensus = self._compute_consensus(shared.responses)
            if consensus >= cfg.consensus_threshold:
                break

            # Conflict resolution
            if cfg.enable_conflict_resolution:
                round_conflicts = self._resolve_conflicts(shared.responses)
                conflicts.extend(round_conflicts)

        # Build final response
        final_response = self._synthesize_final(shared.responses, prompt)
        consensus_score = self._compute_consensus(shared.responses)

        elapsed = (time.perf_counter() - t0) * 1000

        result = SwarmResult(
            final_response=final_response,
            consensus_score=consensus_score,
            rounds_completed=shared.round_number + 1,
            agent_responses=all_responses,
            conflicts=conflicts,
            history=round_history,
            elapsed_ms=elapsed,
        )

        self._history.append({
            "prompt": prompt[:200],
            "consensus": consensus_score,
            "rounds": shared.round_number + 1,
            "elapsed_ms": elapsed,
        })

        return result

    def _execute_round(
        self,
        prompt: str,
        shared: SharedState,
        cfg: SwarmConfig,
    ) -> List[AgentResponse]:
        """Run one round of agent execution."""
        responses: List[AgentResponse] = []

        if cfg.execution_mode == "round_robin":
            # Sequential: each agent sees prior outputs
            for role in cfg.agent_roles:
                agent = self._agents.get(role)
                if agent is None:
                    continue
                resp = agent.process(prompt, shared)
                responses.append(resp)
                shared.responses.append(resp)
        else:
            # Parallel: all agents process independently
            for role in cfg.agent_roles:
                agent = self._agents.get(role)
                if agent is None:
                    continue
                resp = agent.process(prompt, shared)
                responses.append(resp)

        return responses

    def _compute_consensus(self, responses: List[AgentResponse]) -> float:
        """Confidence-weighted consensus score [0, 1]."""
        if not responses:
            return 0.0
        total_weight = sum(r.confidence for r in responses)
        if total_weight == 0:
            return 0.0
        # A simple model: average confidence, penalized by issue flags
        avg_conf = total_weight / len(responses)
        issue_count = sum(
            1 for r in responses
            for f in r.flags
            if f.startswith("issues:") and not f.endswith(":0")
        )
        penalty = min(0.3, issue_count * 0.05)
        return max(0.0, min(1.0, avg_conf - penalty))

    def _resolve_conflicts(self, responses: List[AgentResponse]) -> List[Dict[str, Any]]:
        """Identify and flag conflicts between agent responses."""
        conflicts: List[Dict[str, Any]] = []

        # Check for large confidence gaps between generator and critic
        gen_resps = [r for r in responses if r.role == "generator"]
        critic_resps = [r for r in responses if r.role == "critic"]

        if gen_resps and critic_resps:
            gen_conf = gen_resps[-1].confidence
            crit_conf = critic_resps[-1].confidence
            if abs(gen_conf - crit_conf) > 0.3:
                conflicts.append({
                    "type": "confidence_gap",
                    "agents": ["generator", "critic"],
                    "gap": abs(gen_conf - crit_conf),
                    "resolution": (
                        "Weight toward critic if critic confidence is higher, "
                        "otherwise keep generator output with caveats."
                    ),
                })

        # Check for skeptic concerns
        skeptic_resps = [r for r in responses if r.role == "skeptic"]
        if skeptic_resps:
            last_skeptic = skeptic_resps[-1]
            superlatives = last_skeptic.metadata.get("superlatives", [])
            if len(superlatives) > 3:
                conflicts.append({
                    "type": "overclaiming",
                    "agents": ["skeptic"],
                    "count": len(superlatives),
                    "resolution": "Soften absolute claims in the final response.",
                })

        return conflicts

    def _synthesize_final(
        self,
        responses: List[AgentResponse],
        prompt: str,
    ) -> str:
        """Combine agent outputs into a single final response."""
        # Primary: use the best generator output
        gen_resps = [r for r in responses if r.role == "generator"]
        if not gen_resps:
            return "No response generated."

        best_gen = max(gen_resps, key=lambda r: r.confidence)
        base = best_gen.content

        # Augment with archivist context if relevant
        arch_resps = [r for r in responses if r.role == "archivist" and r.confidence > 0.4]
        if arch_resps:
            ctx = arch_resps[-1].content
            if "Retrieved" in ctx:
                base += f" [Context enriched by archivist.]"

        # Append anomaly warnings
        anomaly_resps = [r for r in responses if r.role == "anomaly_hunter"]
        for ar in anomaly_resps:
            anomalies = ar.metadata.get("anomalies", [])
            if anomalies:
                base += f" [Anomaly note: {anomalies[0]}]"
                break

        return base

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def reset(self) -> None:
        """Reset all agents and history."""
        for agent in self._agents.values():
            agent.reset()
        self._history.clear()

    def __repr__(self) -> str:
        return (
            f"SwarmOrchestrator(agents={list(self._agents.keys())}, "
            f"mode={self.config.execution_mode!r}, "
            f"runs={len(self._history)})"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════

_STOPWORDS: frozenset[str] = frozenset(
    "the a an and or to of in on for with this that is are be as by it from "
    "at we you i they not but if then can could should would about into over "
    "under when what why how yes no".split()
)


def _extract_content_tokens(text: str, max_tokens: int = 50) -> List[str]:
    """Extract meaningful (non-stopword) tokens from text."""
    tokens = re.sub(r"[^a-z0-9\s']", " ", text.lower()).split()
    content = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    return content[:max_tokens]


def _variance(values: Sequence[float]) -> float:
    """Population variance."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)
