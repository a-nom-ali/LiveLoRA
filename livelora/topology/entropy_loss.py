"""Entropy-based TTT loss — the baseline competitor to PH loss.

Implements entropy minimization / confidence sharpening on output
distributions, which is the standard unsupervised signal used in
LoRA-TTT and other test-time adaptation methods.

This serves as method B in the three-way comparison:
  A) No adaptation
  B) Entropy-TTT (this)
  C) PH-TTT (LiveLoRA)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EntropyLoss(nn.Module):
    """Entropy minimization loss on output token distributions.

    Encourages the model to be more confident in its predictions
    by minimizing the entropy of the output distribution.

    This is the classic unsupervised TTT signal — no labels needed,
    just the model's own uncertainty as the optimization target.
    """

    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: How to reduce across tokens — "mean" or "sum".
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy of output distribution.

        Args:
            logits: (batch, seq_len, vocab_size) raw logits from model.

        Returns:
            Scalar entropy loss (lower = more confident).
        """
        # Use last token's logits for causal LM (most relevant for next-token)
        # Or all tokens if we want full-sequence entropy
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # H = -sum(p * log(p))
        entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq_len)

        if self.reduction == "mean":
            return entropy.mean()
        elif self.reduction == "sum":
            return entropy.sum()
        else:
            return entropy


class MarginalEntropyLoss(nn.Module):
    """Marginal entropy loss — encourages diversity across tokens.

    Combines two signals:
    1. Minimize per-token entropy (confidence)
    2. Maximize marginal entropy (diversity across positions)

    This prevents the trivial solution of collapsing to a single
    high-confidence prediction for all positions.
    """

    def __init__(self, diversity_weight: float = 0.1):
        super().__init__()
        self.diversity_weight = diversity_weight

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Per-token entropy (minimize)
        token_entropy = -(probs * log_probs).sum(dim=-1).mean()

        # Marginal entropy across tokens (maximize)
        mean_probs = probs.mean(dim=1)  # (batch, vocab)
        marginal_entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1).mean()

        return token_entropy - self.diversity_weight * marginal_entropy
