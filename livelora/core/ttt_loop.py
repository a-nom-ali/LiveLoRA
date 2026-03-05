"""Test-time training loop: the core LiveLoRA refinement engine.

Implements the per-instance adaptation pattern:
  1. Forward pass through base model + current LoRA
  2. Extract activations at target layers
  3. Compute topological fidelity loss via differentiable PH
  4. Backprop through LoRA parameters only
  5. Update with 1-N gradient steps
  6. Optionally reset LoRA to checkpoint after generation
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from livelora.core.lora_adapter import LiveLoraModel
from livelora.topology.ph_loss import DifferentiablePHLoss


@dataclass
class TTTConfig:
    """Configuration for test-time training loop."""

    num_steps: int = 3
    lr: float = 1e-4
    drift_penalty: float = 0.01
    adapt_and_reset: bool = True
    target_layers: list[int] = field(default_factory=lambda: [-1])
    activation_subsample: int = 64
    grad_clip: float = 1.0


class TTTLoop:
    """Test-time training loop for LiveLoRA.

    Usage:
        ttt = TTTLoop(model, ph_loss, config)
        # Before each input:
        refined_output = ttt.adapt_and_generate(input_ids, attention_mask, generate_kwargs)
    """

    def __init__(
        self,
        model: LiveLoraModel,
        ph_loss: DifferentiablePHLoss,
        config: TTTConfig | None = None,
    ):
        self.model = model
        self.ph_loss = ph_loss
        self.config = config or TTTConfig()

        # Build optimizer over LoRA params only
        self.optimizer = torch.optim.Adam(
            self.model.lora_parameters(),
            lr=self.config.lr,
        )

    def _resolve_layer_indices(self) -> list[int]:
        """Resolve negative layer indices to positive ones."""
        # We need to know the total number of layers; use a heuristic
        # by checking hidden_states length from config
        n_layers = self.model.model.config.num_hidden_layers + 1  # +1 for embedding
        resolved = []
        for idx in self.config.target_layers:
            if idx < 0:
                resolved.append(n_layers + idx)
            else:
                resolved.append(idx)
        return resolved

    def _extract_points(self, activations: dict[int, torch.Tensor]) -> torch.Tensor:
        """Extract point cloud from layer activations for PH computation.

        Takes the first batch element's token activations, subsampled.
        """
        # Concatenate activations from all target layers
        all_points = []
        for layer_idx in sorted(activations.keys()):
            act = activations[layer_idx][0]  # first batch element: (seq_len, hidden_dim)
            all_points.append(act)

        points = torch.cat(all_points, dim=0)  # (total_tokens, hidden_dim)

        # Subsample for tractable PH
        n = points.shape[0]
        if n > self.config.activation_subsample:
            indices = torch.randperm(n, device=points.device)[: self.config.activation_subsample]
            points = points[indices]

        return points

    def refine(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        reference_activations: dict[int, torch.Tensor] | None = None,
    ) -> list[float]:
        """Run test-time refinement steps on current input.

        Args:
            input_ids: (1, seq_len) input token IDs.
            attention_mask: Optional attention mask.
            reference_activations: Optional base model activations for divergence loss.

        Returns:
            List of loss values per step (for monitoring).
        """
        layer_indices = self._resolve_layer_indices()
        losses = []

        for step in range(self.config.num_steps):
            self.optimizer.zero_grad()

            # Forward pass to get activations
            activations = self.model.get_layer_activations(
                input_ids=input_ids,
                attention_mask=attention_mask,
                layer_indices=layer_indices,
            )

            # Extract point cloud for PH
            points = self._extract_points(activations)

            # Compute topological loss
            ref_points = None
            if reference_activations is not None:
                ref_points = self._extract_points(reference_activations)

            topo_loss = self.ph_loss(points, reference_activations=ref_points)

            # Add drift regularization
            drift = self.model.lora_l2_from_checkpoint()
            total_loss = topo_loss + self.config.drift_penalty * drift

            # Backward + update
            total_loss.backward()
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.lora_parameters(),
                    self.config.grad_clip,
                )
            self.optimizer.step()

            losses.append(total_loss.item())

        return losses

    def adapt_and_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        generate_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, list[float]]:
        """Full adapt-and-generate cycle.

        1. Checkpoint LoRA state
        2. Run refinement steps
        3. Generate output
        4. Optionally restore LoRA state

        Returns:
            (generated_ids, losses) tuple.
        """
        generate_kwargs = generate_kwargs or {}

        # Save state
        self.model.checkpoint()

        # Refine
        self.model.train()
        losses = self.refine(input_ids, attention_mask)

        # Generate
        self.model.eval()
        with torch.no_grad():
            output = self.model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generate_kwargs,
            )

        # Reset if configured
        if self.config.adapt_and_reset:
            self.model.restore()

        return output, losses
