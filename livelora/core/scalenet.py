"""ScaleNet: per-layer, per-step learning rate modulation.

A small hypernetwork that predicts optimal learning rate multipliers
for each LoRA layer based on the current topological loss signal.
Inspired by the Unsupervised Dynamic TTA paper (arXiv:2602.09719).

This is the "surprise-proportional" component: layers that contribute
more to topological error get larger learning rates.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ScaleNet(nn.Module):
    """Predicts per-layer LR scale factors from loss statistics.

    Input: a feature vector summarizing the current topological state
        (e.g., loss value, gradient norms per layer, persistence statistics).
    Output: a scale factor per LoRA layer (multiplied with base LR).
    """

    def __init__(self, num_layers: int, input_dim: int = 16, hidden_dim: int = 32):
        """
        Args:
            num_layers: Number of LoRA layer groups to modulate.
            input_dim: Dimension of the input signal vector.
            hidden_dim: Hidden layer size.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_layers),
            nn.Softplus(),  # Ensure positive scale factors
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signal: (input_dim,) tensor with topological signal features.

        Returns:
            (num_layers,) positive scale factors.
        """
        return self.net(signal)

    @staticmethod
    def build_signal(
        loss_value: float,
        grad_norms: list[float],
        persistence_stats: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """Build the input signal vector from current training state.

        This creates a fixed-size feature vector from variable-length inputs
        by using summary statistics.
        """
        features = [loss_value]

        # Gradient norm statistics
        if grad_norms:
            t = torch.tensor(grad_norms)
            features.extend([
                t.mean().item(),
                t.std().item() if len(t) > 1 else 0.0,
                t.max().item(),
                t.min().item(),
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        # Persistence statistics
        if persistence_stats:
            features.extend([
                persistence_stats.get("total_persistence", 0.0),
                persistence_stats.get("num_features_h0", 0.0),
                persistence_stats.get("num_features_h1", 0.0),
                persistence_stats.get("max_persistence", 0.0),
                persistence_stats.get("mean_persistence", 0.0),
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # Pad to fixed size
        features.extend([0.0] * max(0, 16 - len(features)))

        return torch.tensor(features[:16], dtype=torch.float32)
