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
from livelora.topology.entropy_loss import EntropyLoss
from livelora.topology.ph_loss import DifferentiablePHLoss, _activations_to_distance_matrix
from livelora.topology.ph_tracker import PHTracker, TopologyState, _deterministic_subsample


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
    kl_probe_len: int = 8  # Number of positions for KL trust region

    # Stability
    cooldown_chunks: int = 1  # Min chunks between updates
    max_updates: int = 10  # Max updates per generation

    # Conditional PH escalation
    conditional_ph: bool = True  # Enable topology-state-gated PH
    max_attempts_drifting: int = 1  # Max update attempts when DRIFTING
    max_attempts_collapsing: int = 2  # Max update attempts when COLLAPSING

    # State-dependent gate thresholds (COLLAPSING is more permissive)
    state_dependent_gate: bool = True
    epsilon_kl_collapsing: float = 5e-3  # Looser KL for COLLAPSING
    tau_rho_collapsing: float = 0.0  # Lower rho bar for COLLAPSING (accept more)

    # Optimization mode: "ph", "entropy", or "hybrid"
    #   ph      — optimize PH loss (original LiveLoRA)
    #   entropy — optimize entropy loss when topology triggers (PH→Entropy)
    #   hybrid  — optimize alpha*entropy + beta*topo + drift (best in Phase 1)
    optimization_mode: str = "hybrid"
    alpha_entropy: float = 1.0  # Weight for entropy term (hybrid/entropy modes)
    beta_topo: float = 0.01  # Weight for PH term (hybrid mode only)
    entropy_probe_len: int = 8  # Positions to compute entropy over (tail tokens)
    divergence_drift_threshold: float = 1.5  # Absolute divergence to trigger DRIFTING


@dataclass
class ChunkMetrics:
    """Metrics from a single chunk evaluation.

    Stored for EVERY chunk, even when no update is attempted,
    so that chunk_idx is always the real chunk number.
    """

    chunk_idx: int
    topology_state: str
    attempts: int = 0  # How many update attempts were made
    reason: str = ""  # "stable_skip", "cooldown", "max_updates", "rejected_kl", "rejected_rho", "accepted"
    topo_loss_before: float = 0.0
    topo_loss_after: float = 0.0
    drift_before: float = 0.0
    drift_after: float = 0.0
    struct_loss_before: float = 0.0
    struct_loss_after: float = 0.0
    kl_divergence: float = 0.0
    delta_struct: float = 0.0
    rho: float = 0.0
    accepted: bool = False
    eff_rank: float = 0.0
    cos_conc: float = 0.0
    divergence: float = 0.0
    topo_improved: bool = False  # Did topology divergence decrease after update?


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
        )
        self.entropy_loss = EntropyLoss(reduction="mean")
        self.tracker = PHTracker(
            max_points=self.config.max_points,
            max_dimension=self.config.max_dimension,
            divergence_drift_threshold=self.config.divergence_drift_threshold,
        )
        # Persistent optimizer (avoids re-creation per candidate step)
        self._optimizer = torch.optim.SGD(
            self.model.lora_parameters(), lr=self.config.lr,
        )

    def _resolve_layer_indices(self) -> list[int]:
        """Resolve negative layer indices."""
        n_layers = self.model.model.config.num_hidden_layers + 1
        return [n_layers + idx if idx < 0 else idx for idx in self.config.target_layers]

    def _get_points(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get activation point cloud from target layers (no logits).

        Uses deterministic subsampling for stable topology comparisons.
        """
        layer_indices = self._resolve_layer_indices()
        activations = self.model.get_layer_activations(
            input_ids=input_ids,
            attention_mask=attention_mask,
            layer_indices=layer_indices,
        )
        all_points = []
        for idx in sorted(activations.keys()):
            all_points.append(activations[idx][0])  # (seq_len, hidden_dim)
        points = torch.cat(all_points, dim=0)

        return _deterministic_subsample(points, self.config.max_points)

    def _get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass returning logits only (no hidden_states overhead)."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def _get_activations(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (logits, activation_points).

        Convenience method when both are needed.
        """
        points = self._get_points(input_ids, attention_mask)
        logits = self._get_logits(input_ids, attention_mask)
        return logits, points

    @staticmethod
    def _kl_trust_region(
        logits_before: torch.Tensor,
        logits_after: torch.Tensor,
        probe_len: int = 8,
        eps: float = 1e-12,
    ) -> float:
        """KL divergence averaged over last probe_len token positions.

        Constrains semantic drift across the chunk, not just the last token.
        """
        seq_len = logits_before.shape[1]
        k = min(probe_len, seq_len)
        probe_idx = torch.arange(seq_len - k, seq_len, device=logits_before.device)

        lb = logits_before[:, probe_idx, :]
        la = logits_after[:, probe_idx, :]

        p = F.softmax(lb, dim=-1).clamp_min(eps)
        logq = F.log_softmax(la, dim=-1)

        kl = (p * (p.log() - logq)).sum(dim=-1)  # (batch, k)
        return float(kl.mean().item())

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

    def _compute_optimization_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute the loss to backprop through, based on optimization_mode.

        Returns a scalar loss tensor with gradients.
        """
        cfg = self.config
        mode = cfg.optimization_mode

        if mode == "ph":
            points = self._get_points(input_ids, attention_mask)
            struct_loss, _ = self._compute_struct_loss(points)
            return struct_loss

        elif mode == "entropy":
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            # Entropy over tail tokens (most relevant for next-token prediction)
            k = min(cfg.entropy_probe_len, logits.shape[1])
            tail_logits = logits[:, -k:, :]
            ent_loss = self.entropy_loss(tail_logits)
            drift = self.model.lora_l2_from_checkpoint()
            return cfg.alpha_entropy * ent_loss + cfg.lambda_drift * drift

        elif mode == "hybrid":
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            logits = outputs.logits
            # Entropy component
            k = min(cfg.entropy_probe_len, logits.shape[1])
            tail_logits = logits[:, -k:, :]
            ent_loss = self.entropy_loss(tail_logits)
            # PH component from activations
            layer_indices = self._resolve_layer_indices()
            hidden_states = outputs.hidden_states
            all_points = []
            for idx in sorted(layer_indices):
                all_points.append(hidden_states[idx][0])
            points = torch.cat(all_points, dim=0)
            points = _deterministic_subsample(points, cfg.max_points)
            topo_loss = self.ph_loss(points)
            drift = self.model.lora_l2_from_checkpoint()
            return (
                cfg.alpha_entropy * ent_loss
                + cfg.beta_topo * topo_loss
                + cfg.lambda_drift * drift
            )

        else:
            raise ValueError(f"Unknown optimization_mode: {mode!r}")

    def _try_update(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        chunk_idx: int,
        topo_state: TopologyState,
    ) -> ChunkMetrics:
        """Attempt one candidate LoRA update with MDL ratio gate.

        Steps:
          1. Measure baseline structural + semantic state
          2. Do one gradient step on LoRA (using configured optimization_mode)
          3. Measure post-update state
          4. Accept or reject based on KL trust + topology-aware gate

        Gate logic depends on optimization_mode:
          - "ph": accept if structural loss improves (rho = d_struct / d_sem)
          - "entropy"/"hybrid": accept if topology divergence improves
            (rho_ctl = d_topo_div / d_sem) — the optimizer uses entropy
            but acceptance is gated on topology improvement
        """
        cfg = self.config

        # (0) Baseline: logits (for KL), points (for struct measurement), topo divergence
        with torch.no_grad():
            logits_before = self._get_logits(input_ids, attention_mask).detach()
            points_before = self._get_points(input_ids, attention_mask).detach()

        struct_before, topo_before = self._compute_struct_loss(points_before)
        drift_before = self.model.lora_l2_from_checkpoint()
        div_before = self.tracker.divergence_from_baseline()

        # (1) Candidate update: one gradient step using configured loss
        self.model.train()
        candidate_checkpoint = {
            name: param.data.clone()
            for name, param in self.model.model.named_parameters()
            if "lora_" in name
        }

        opt_loss = self._compute_optimization_loss(input_ids, attention_mask)

        self._optimizer.zero_grad()
        opt_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.lora_parameters(), cfg.grad_clip)
        self._optimizer.step()

        # (2) Evaluate candidate
        self.model.eval()
        with torch.no_grad():
            logits_after = self._get_logits(input_ids, attention_mask)
            points_after = self._get_points(input_ids, attention_mask)

            struct_after, topo_after = self._compute_struct_loss(points_after)
            drift_after = self.model.lora_l2_from_checkpoint()

            # Re-observe topology to get post-update divergence
            obs_after = self.tracker._compute_summary(points_after)
            # Temporarily check divergence without modifying history
            div_after = abs(
                obs_after.total_persistence - self.tracker._baseline.total_persistence
            ) / max(self.tracker._baseline.total_persistence, 1e-12) if self.tracker._baseline else 0.0

            # Multi-position KL
            kl = self._kl_trust_region(
                logits_before, logits_after, probe_len=cfg.kl_probe_len,
            )

            d_struct = float(struct_before.item()) - float(struct_after.item())
            d_sem = max(kl, 0.0)

            # Topology divergence change
            d_topo_div = div_before - div_after  # positive = divergence decreased = good

            # Gate logic: depends on optimization mode
            if cfg.optimization_mode == "ph":
                # Original: rho = structural improvement / semantic cost
                rho = d_struct / (d_sem + cfg.beta)
                gate_improvement = d_struct > 0
            else:
                # Entropy/hybrid: accept if topology divergence improved OR structural loss improved
                # Entropy optimization improves output quality even when divergence doesn't drop
                rho = d_topo_div / (d_sem + cfg.beta) if d_topo_div > 0 else d_struct / (d_sem + cfg.beta)
                gate_improvement = d_topo_div > 0 or d_struct > 0

        # (3) Accept / reject with state-dependent thresholds
        if cfg.state_dependent_gate and topo_state == TopologyState.COLLAPSING:
            effective_kl = cfg.epsilon_kl_collapsing
            effective_tau = cfg.tau_rho_collapsing
        else:
            effective_kl = cfg.epsilon_kl
            effective_tau = cfg.tau_rho

        if kl > effective_kl:
            accept = False
            reason = "rejected_kl"
        elif rho < effective_tau:
            accept = False
            reason = "rejected_rho"
        elif not gate_improvement:
            accept = False
            reason = "rejected_no_improvement"
        else:
            accept = True
            reason = "accepted"

        if not accept:
            for name, param in self.model.model.named_parameters():
                if "lora_" in name and name in candidate_checkpoint:
                    param.data.copy_(candidate_checkpoint[name])

        # Get proxy values from latest tracker observation
        latest = self.tracker._history[-1] if self.tracker._history else None

        return ChunkMetrics(
            chunk_idx=chunk_idx,
            topology_state=topo_state.value,
            attempts=1,
            reason=reason,
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
            eff_rank=latest.eff_rank if latest else 0.0,
            cos_conc=latest.cos_conc if latest else 0.0,
            divergence=self.tracker.divergence_from_baseline(),
            topo_improved=d_topo_div > 0,
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
            Metrics list has exactly one entry per chunk (aligned by chunk_idx).
        """
        cfg = self.config

        # Checkpoint LoRA at start
        self.model.checkpoint()
        self.tracker.reset()

        # Set topology baseline from prompt activations (points only — no logits needed)
        self.model.eval()
        with torch.no_grad():
            baseline_points = self._get_points(input_ids, attention_mask)
            self.tracker.set_baseline(baseline_points)

        all_metrics: list[ChunkMetrics] = []
        current_ids = input_ids
        current_mask = attention_mask
        chunks_since_update = cfg.cooldown_chunks  # Allow first update
        total_updates = 0
        total_generated = 0
        chunk_idx = 0

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

            # Observe topology (points only — cheap, no logits)
            with torch.no_grad():
                obs_points = self._get_points(current_ids, current_mask)
                obs_summary = self.tracker.observe(obs_points)
                topo_state = self.tracker.assess()

            # Decide whether to attempt update
            skip_reason = None
            if chunks_since_update < cfg.cooldown_chunks:
                skip_reason = "cooldown"
            elif total_updates >= cfg.max_updates:
                skip_reason = "max_updates"
            elif cfg.conditional_ph and topo_state == TopologyState.STABLE:
                skip_reason = "stable_skip"

            if skip_reason is not None:
                # Record a no-attempt metric for this chunk
                all_metrics.append(ChunkMetrics(
                    chunk_idx=chunk_idx,
                    topology_state=topo_state.value,
                    attempts=0,
                    reason=skip_reason,
                    eff_rank=obs_summary.eff_rank,
                    cos_conc=obs_summary.cos_conc,
                    divergence=self.tracker.divergence_from_baseline(),
                ))
                chunks_since_update += 1
            else:
                # Determine max attempts based on escalation
                if cfg.conditional_ph:
                    max_attempts = (
                        cfg.max_attempts_collapsing if topo_state == TopologyState.COLLAPSING
                        else cfg.max_attempts_drifting
                    )
                else:
                    max_attempts = 1

                best_metrics = None
                for attempt in range(max_attempts):
                    metrics = self._try_update(current_ids, current_mask, chunk_idx, topo_state)
                    best_metrics = metrics

                    if metrics.accepted:
                        total_updates += 1
                        chunks_since_update = 0
                        break
                else:
                    chunks_since_update += 1

                if best_metrics is not None:
                    best_metrics.attempts = min(attempt + 1, max_attempts) if max_attempts > 0 else 0
                    all_metrics.append(best_metrics)

            chunk_idx += 1

        return current_ids, all_metrics
