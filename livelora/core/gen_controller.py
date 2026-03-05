"""LiveLoRA-Delta: chunked generation with MDL ratio gate.

Generates text in chunks, evaluating topology at each boundary and
applying LoRA updates only when they pass the accept/reject gate:
  1. KL trust region (semantic pinning)
  2. Rho payoff gate (structural gain >> semantic drift)
  3. Net structural improvement

This prevents the adapter from "learning the conversation" and
keeps updates as topology repair / structural compression.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from livelora.core.lora_adapter import LiveLoraModel
from livelora.topology.ph_loss import DifferentiablePHLoss, _activations_to_distance_matrix
from livelora.topology.ph_tracker import PHTracker, TopologyState


@dataclass
class DeltaConfig:
    """Configuration for LiveLoRA-Delta chunked generation."""

    chunk_size: int = 32
    max_new_tokens: int = 256
    target_layers: list[int] = field(default_factory=lambda: [-1])

    # PH settings
    max_points: int = 64
    max_dimension: int = 1

    # MDL ratio gate parameters
    lr: float = 1e-4
    grad_clip: float = 1.0
    lambda_drift: float = 1e-2  # L2 drift penalty weight
    epsilon_kl: float = 1e-3  # KL trust region bound
    tau_rho: float = 50.0  # Minimum rho for acceptance
    beta: float = 1e-8  # Division safety constant

    # Stability
    cooldown_chunks: int = 1  # Min chunks between updates
    max_updates: int = 10  # Max updates per generation

    # Conditional PH escalation
    conditional_ph: bool = True  # Enable topology-state-gated PH
    max_attempts_drifting: int = 1  # Max update attempts when DRIFTING
    max_attempts_collapsing: int = 2  # Max update attempts when COLLAPSING


@dataclass
class ChunkMetrics:
    """Metrics from a single chunk evaluation."""

    chunk_idx: int
    topo_loss_before: float
    topo_loss_after: float
    drift_before: float
    drift_after: float
    struct_loss_before: float
    struct_loss_after: float
    kl_divergence: float
    delta_struct: float
    rho: float
    accepted: bool
    topology_state: str


class GenerationController:
    """Chunked generation with topology-gated LoRA updates.

    Usage:
        controller = GenerationController(model, config)
        output_ids, metrics = controller.generate(input_ids, attention_mask)
    """

    def __init__(
        self,
        model: LiveLoraModel,
        config: DeltaConfig | None = None,
    ):
        self.model = model
        self.config = config or DeltaConfig()
        self.ph_loss = DifferentiablePHLoss(
            max_dimension=self.config.max_dimension,
            max_points=self.config.max_points,
            target_betti={0: 1, 1: 0},
        )
        self.tracker = PHTracker(
            max_points=self.config.max_points,
            max_dimension=self.config.max_dimension,
        )

    def _resolve_layer_indices(self) -> list[int]:
        """Resolve negative layer indices."""
        n_layers = self.model.model.config.num_hidden_layers + 1
        return [n_layers + idx if idx < 0 else idx for idx in self.config.target_layers]

    def _get_activations(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (logits, activation_points).

        activation_points: (n_tokens, hidden_dim) from target layers.
        """
        layer_indices = self._resolve_layer_indices()
        activations = self.model.get_layer_activations(
            input_ids=input_ids,
            attention_mask=attention_mask,
            layer_indices=layer_indices,
        )
        # Concatenate activations from target layers, first batch element
        all_points = []
        for idx in sorted(activations.keys()):
            all_points.append(activations[idx][0])  # (seq_len, hidden_dim)
        points = torch.cat(all_points, dim=0)

        # Subsample
        if points.shape[0] > self.config.max_points:
            indices = torch.randperm(points.shape[0], device=points.device)[
                : self.config.max_points
            ]
            points = points[indices]

        # Get logits from a regular forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits, points

    def _compute_struct_loss(
        self,
        points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute structural loss = topo_distance + lambda * drift.

        Returns (struct_loss, topo_loss) — topo_loss for logging.
        """
        topo_loss = self.ph_loss(points)
        drift = self.model.lora_l2_from_checkpoint()
        struct_loss = topo_loss + self.config.lambda_drift * drift
        return struct_loss, topo_loss

    def _try_update(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        chunk_idx: int,
    ) -> ChunkMetrics:
        """Attempt one candidate LoRA update with MDL ratio gate.

        Steps:
          1. Measure baseline structural + semantic state
          2. Do one gradient step on LoRA
          3. Measure post-update state
          4. Accept or reject based on KL trust + rho gate + improvement
        """
        cfg = self.config

        # (0) Baseline forward (no gradients for this measurement)
        with torch.no_grad():
            logits_before, points_before = self._get_activations(input_ids, attention_mask)
            # Detach logits for KL reference
            p_before = F.softmax(logits_before[:, -1, :], dim=-1).detach()

        # Compute baseline structural loss (needs grad for topo)
        struct_before, topo_before = self._compute_struct_loss(points_before.detach())
        drift_before = self.model.lora_l2_from_checkpoint()

        # (1) Candidate update: one gradient step
        self.model.train()
        # Save candidate state for potential rollback
        candidate_checkpoint = {
            name: param.data.clone()
            for name, param in self.model.model.named_parameters()
            if "lora_" in name
        }

        # Forward with gradients
        _, points_grad = self._get_activations(input_ids, attention_mask)
        struct_loss, _ = self._compute_struct_loss(points_grad)

        # Gradient step
        optimizer = torch.optim.SGD(self.model.lora_parameters(), lr=cfg.lr)
        optimizer.zero_grad()
        struct_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.lora_parameters(), cfg.grad_clip)
        optimizer.step()

        # (2) Evaluate candidate
        self.model.eval()
        with torch.no_grad():
            logits_after, points_after = self._get_activations(input_ids, attention_mask)
            p_after = F.softmax(logits_after[:, -1, :], dim=-1)

            struct_after, topo_after = self._compute_struct_loss(points_after)
            drift_after = self.model.lora_l2_from_checkpoint()

            # KL divergence (semantic displacement)
            # Clamp for numerical stability
            kl = F.kl_div(
                torch.log(p_after.clamp(min=1e-10)),
                p_before,
                reduction="batchmean",
                log_target=False,
            ).item()

            d_struct = float(struct_before.item()) - float(struct_after.item())
            d_sem = max(kl, 0.0)
            rho = d_struct / (d_sem + cfg.beta)

        # (3) Accept / reject
        accept = (kl <= cfg.epsilon_kl) and (rho >= cfg.tau_rho) and (d_struct > 0)

        if not accept:
            # Rollback
            for name, param in self.model.model.named_parameters():
                if "lora_" in name and name in candidate_checkpoint:
                    param.data.copy_(candidate_checkpoint[name])

        # Use last assessed state from the generate loop
        state = self.tracker.assess()

        return ChunkMetrics(
            chunk_idx=chunk_idx,
            topo_loss_before=float(topo_before.item()),
            topo_loss_after=float(topo_after.item()),
            drift_before=float(drift_before.item()) if isinstance(drift_before, torch.Tensor) else float(drift_before),
            drift_after=float(drift_after.item()) if isinstance(drift_after, torch.Tensor) else float(drift_after),
            struct_loss_before=float(struct_before.item()),
            struct_loss_after=float(struct_after.item()),
            kl_divergence=kl,
            delta_struct=d_struct,
            rho=rho,
            accepted=accept,
            topology_state=state.value,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **generate_kwargs,
    ) -> tuple[torch.Tensor, list[ChunkMetrics]]:
        """Generate text with chunked topology-gated LoRA adaptation.

        Args:
            input_ids: (1, seq_len) prompt tokens.
            attention_mask: Optional mask.
            **generate_kwargs: Passed to model.generate() per chunk.

        Returns:
            (generated_ids, list of ChunkMetrics per chunk).
        """
        cfg = self.config

        # Checkpoint LoRA at start
        self.model.checkpoint()
        self.tracker.reset()

        # Set topology baseline from prompt activations
        self.model.eval()
        with torch.no_grad():
            _, baseline_points = self._get_activations(input_ids, attention_mask)
            self.tracker.set_baseline(baseline_points)

        all_metrics: list[ChunkMetrics] = []
        current_ids = input_ids
        current_mask = attention_mask
        chunks_since_update = cfg.cooldown_chunks  # Allow first update
        total_updates = 0
        total_generated = 0

        while total_generated < cfg.max_new_tokens:
            tokens_to_gen = min(cfg.chunk_size, cfg.max_new_tokens - total_generated)

            # Generate one chunk
            self.model.eval()
            with torch.no_grad():
                gen_output = self.model.model.generate(
                    input_ids=current_ids,
                    attention_mask=current_mask,
                    max_new_tokens=tokens_to_gen,
                    do_sample=generate_kwargs.get("do_sample", False),
                    temperature=generate_kwargs.get("temperature", 1.0),
                    pad_token_id=generate_kwargs.get("pad_token_id"),
                )

            # Update current sequence
            new_tokens = gen_output.shape[1] - current_ids.shape[1]
            if new_tokens == 0:
                break  # EOS or generation stopped

            current_ids = gen_output
            if current_mask is not None:
                current_mask = torch.cat(
                    [current_mask, torch.ones(1, new_tokens, device=current_mask.device, dtype=current_mask.dtype)],
                    dim=1,
                )
            total_generated += new_tokens
            chunk_idx = len(all_metrics)

            # Check topology state before deciding whether to attempt update
            can_update = (
                chunks_since_update >= cfg.cooldown_chunks
                and total_updates < cfg.max_updates
            )

            if can_update:
                # Observe current topology (cheap — just PH computation, no grad)
                with torch.no_grad():
                    _, obs_points = self._get_activations(current_ids, current_mask)
                    self.tracker.observe(obs_points)
                    topo_state = self.tracker.assess()

                if cfg.conditional_ph:
                    # Escalation: decide update attempts based on topology state
                    if topo_state == TopologyState.STABLE:
                        # Topology is fine — skip PH update entirely
                        attempts = 0
                    elif topo_state == TopologyState.DRIFTING:
                        attempts = cfg.max_attempts_drifting
                    else:  # COLLAPSING
                        attempts = cfg.max_attempts_collapsing
                else:
                    # No escalation — always try once (original behavior)
                    attempts = 1

                for attempt in range(attempts):
                    metrics = self._try_update(current_ids, current_mask, chunk_idx)
                    all_metrics.append(metrics)

                    if metrics.accepted:
                        total_updates += 1
                        chunks_since_update = 0
                        break
                else:
                    # No attempts or all rejected
                    chunks_since_update += 1
            else:
                chunks_since_update += 1

        return current_ids, all_metrics
