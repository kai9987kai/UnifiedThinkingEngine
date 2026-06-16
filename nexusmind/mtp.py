"""Multi-Token Prediction head (from Xiaomi MiMo).

Predicts multiple future tokens simultaneously for speculative decoding.
Each head shares the backbone representation but makes independent predictions.
Training loss = sum of cross-entropy for each prediction head.
During inference: predict N tokens at once, verify with main head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MTPHead(nn.Module):
    """Single future-token prediction head.

    Projects hidden states to vocab logits for predicting the i-th next token.
    Uses a small refinement MLP before the final projection to allow each head
    to learn position-specific adjustments.
    """

    def __init__(self, d_model: int, vocab_size: int, head_idx: int):
        super().__init__()
        self.head_idx = head_idx
        self.refine = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        # Position offset embedding — tells the head which future position it predicts
        self.pos_offset = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden: (batch, seq, d_model) backbone hidden states.

        Returns:
            logits: (batch, seq, vocab_size) predictions for the (head_idx+1)-th next token.
        """
        x = hidden + self.pos_offset
        x = self.refine(x)
        return self.proj(x)


class MultiTokenPredictionHead(nn.Module):
    """Multi-Token Prediction module from Xiaomi MiMo.

    Predicts N future tokens simultaneously from the same backbone representation.
    This enables speculative decoding at inference time: generate N candidates in
    one forward pass, then verify each with the main LM head.

    Key benefits (from MiMo paper):
    - ~2x inference speedup via speculative decoding
    - Better representation learning during training (multi-task regularization)
    - Forces the backbone to encode richer future-oriented representations

    Architecture:
        backbone_hidden → MTPHead_1 → logits for token t+1
                        → MTPHead_2 → logits for token t+2
                        → ...
                        → MTPHead_N → logits for token t+N
    """

    def __init__(self, d_model: int, vocab_size: int, n_heads: int = 2):
        super().__init__()
        self.n_heads = n_heads
        self.heads = nn.ModuleList([
            MTPHead(d_model, vocab_size, i) for i in range(n_heads)
        ])

    def forward(
        self, hidden: torch.Tensor, target_ids: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Forward pass with optional loss computation.

        Args:
            hidden: (batch, seq, d_model) backbone hidden states.
            target_ids: (batch, seq) ground-truth token ids for loss computation.
                        If provided, computes cross-entropy loss for each head.

        Returns:
            dict with keys:
                'logits': list of (batch, seq, vocab_size) per head
                'loss': scalar MTP loss (only if target_ids provided)
                'predictions': list of (batch, seq) argmax token ids per head
        """
        all_logits = [head(hidden) for head in self.heads]
        predictions = [logits.argmax(dim=-1) for logits in all_logits]

        result = {
            "logits": all_logits,
            "predictions": predictions,
        }

        if target_ids is not None:
            total_loss = torch.tensor(0.0, device=hidden.device)
            for i, logits in enumerate(all_logits):
                # Head i predicts token at position t + (i+1)
                # Shift targets by (i+1) positions
                shift = i + 1
                if target_ids.size(1) > shift:
                    shifted_targets = target_ids[:, shift:]  # (B, S - shift)
                    shifted_logits = logits[:, :-shift, :]    # (B, S - shift, V)
                    loss_i = F.cross_entropy(
                        shifted_logits.reshape(-1, shifted_logits.size(-1)),
                        shifted_targets.reshape(-1),
                        ignore_index=-100,
                    )
                    total_loss = total_loss + loss_i
            result["loss"] = total_loss / max(len(all_logits), 1)

        return result

    def speculative_decode(
        self, hidden_last: torch.Tensor
    ) -> list[torch.Tensor]:
        """Speculative decoding: predict N future tokens from the last hidden state.

        Args:
            hidden_last: (batch, 1, d_model) hidden state of the last token.

        Returns:
            List of (batch, 1) predicted token ids for positions t+1, t+2, ..., t+N.
        """
        candidates = []
        for head in self.heads:
            logits = head(hidden_last)  # (B, 1, V)
            token = logits.argmax(dim=-1)  # (B, 1)
            candidates.append(token)
        return candidates


class MTPLoss(nn.Module):
    """Standalone MTP loss module for use in training loops.

    Computes weighted cross-entropy across all MTP heads.
    Heads predicting further into the future get lower weight
    (geometric decay) since accuracy naturally decreases.
    """

    def __init__(self, n_heads: int, decay: float = 0.8):
        super().__init__()
        self.n_heads = n_heads
        self.decay = decay
        weights = [decay ** i for i in range(n_heads)]
        total = sum(weights)
        self.register_buffer(
            "head_weights",
            torch.tensor([w / total for w in weights]),
        )

    def forward(
        self, mtp_logits: list[torch.Tensor], target_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted MTP loss.

        Args:
            mtp_logits: list of (batch, seq, vocab_size) per head.
            target_ids: (batch, seq) ground-truth token ids.

        Returns:
            Scalar loss.
        """
        total_loss = torch.tensor(0.0, device=target_ids.device)
        for i, logits in enumerate(mtp_logits):
            shift = i + 1
            if target_ids.size(1) > shift:
                shifted_targets = target_ids[:, shift:]
                shifted_logits = logits[:, :-shift, :]
                loss_i = F.cross_entropy(
                    shifted_logits.reshape(-1, shifted_logits.size(-1)),
                    shifted_targets.reshape(-1),
                    ignore_index=-100,
                )
                total_loss = total_loss + self.head_weights[i] * loss_i
        return total_loss
