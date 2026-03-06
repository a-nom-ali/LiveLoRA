"""Rolling topology baseline tracker for LiveLoRA-Delta.

Maintains a running summary of topological features across generation chunks,
detects when topology destabilizes, and provides the self-consistency target
(pi-star) for the MDL ratio gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch

from livelora.topology.ph_loss import DifferentiablePHLoss, _activations_to_distance_matrix


class TopologyState(Enum):
    """Assessment of current topological stability."""

    STABLE = "stable"  # Topology close to baseline — skip update
    DRIFTING = "drifting"  # Topology diverging — consider small update
    COLLAPSING = "collapsing"  # Topology severely degraded — update urgently


@dataclass
class TopologySummary:
    """Compact summary of a persistence diagram + cheap proxies."""

    betti_0: int = 0  # Number of connected components above threshold
    betti_1: int = 0  # Number of loops above threshold
    total_persistence: float = 0.0  # Sum of all (death - birth) values
    max_persistence: float = 0.0  # Largest single feature persistence
    mean_persistence: float = 0.0  # Average persistence
    num_features: int = 0  # Total features above threshold

    # Cheap proxies (computed before PH, useful for early collapse detection)
    eff_rank: float = 0.0  # Effective rank of activation matrix
    cos_conc: float = 0.0  # Mean absolute cosine similarity (anisotropy)


def effective_rank(X: torch.Tensor, eps: float = 1e-12) -> float:
    """Effective rank via entropy of normalized singular values.

    Low effective rank => dimensional collapse (tokens in a low-dim subspace).
    Cheap for N <= 64: computes N x N covariance eigenvalues.
    """
    X = X.float()  # eigvalsh requires float32 on CUDA
    X = X - X.mean(dim=0, keepdim=True)
    C = (X @ X.T) / max(X.shape[1], 1)
    evals = torch.linalg.eigvalsh(C).clamp_min(eps)
    p = evals / evals.sum()
    H = -(p * torch.log(p)).sum()
    return float(torch.exp(H).item())


def mean_abs_cosine(X: torch.Tensor, eps: float = 1e-12) -> float:
    """Mean absolute cosine similarity between all token pairs.

    High value => tokens too aligned => collapse / attractor dominance.
    """
    X = X.float()  # norm computation needs float32 on CUDA
    Xn = X / (X.norm(dim=1, keepdim=True) + eps)
    S = Xn @ Xn.T
    n = S.shape[0]
    if n < 2:
        return 0.0
    return float((S.abs().sum() - n) / (n * (n - 1) + eps))


def _deterministic_subsample(activations: torch.Tensor, max_points: int) -> torch.Tensor:
    """Deterministic strided subsampling (no randomness => stable comparisons)."""
    if activations.shape[0] <= max_points:
        return activations
    indices = torch.linspace(0, activations.shape[0] - 1, steps=max_points).long()
    return activations[indices.to(activations.device)]


class PHTracker:
    """Track topology over generation chunks and detect destabilization.

    Maintains a rolling baseline of TopologySummary stats and compares
    each new chunk against it to classify the topology state.

    Usage:
        tracker = PHTracker(max_points=64)

        # At checkpoint (start of generation):
        tracker.set_baseline(initial_activations)

        # Per chunk:
        summary = tracker.observe(chunk_activations)
        state = tracker.assess()
        if state != TopologyState.STABLE:
            # trigger LoRA update
    """

    def __init__(
        self,
        max_points: int = 64,
        max_dimension: int = 1,
        persistence_threshold: float = 0.01,
        drift_threshold: float = 0.3,
        collapse_threshold: float = 0.7,
        window_size: int = 5,
        baseline_mode: str = "fixed",
        ema_alpha: float = 0.05,
        divergence_drift_threshold: float = 1.5,
    ):
        """
        Args:
            max_points: Subsample activation points to this count.
            max_dimension: Max homological dimension for PH.
            persistence_threshold: Min persistence to count a feature.
            drift_threshold: Relative change from baseline to flag drifting.
            collapse_threshold: Relative change from baseline to flag collapse.
            window_size: Number of recent summaries to keep for rolling stats.
            baseline_mode: "fixed" (frozen at checkpoint) or "ema" (slow follow).
            ema_alpha: EMA update rate when baseline_mode="ema" and state is STABLE.
            divergence_drift_threshold: Absolute divergence from baseline to flag drifting.
                Catches topology expansion (not just degradation). Set to inf to disable.
        """
        self.max_points = max_points
        self.max_dimension = max_dimension
        self.persistence_threshold = persistence_threshold
        self.drift_threshold = drift_threshold
        self.collapse_threshold = collapse_threshold
        self.window_size = window_size
        self.baseline_mode = baseline_mode
        self.ema_alpha = ema_alpha
        self.divergence_drift_threshold = divergence_drift_threshold

        self._baseline: TopologySummary | None = None
        self._history: list[TopologySummary] = []
        self._ph_loss = DifferentiablePHLoss(
            max_dimension=max_dimension,
            max_points=max_points,
            persistence_threshold=persistence_threshold,
        )

    def _compute_summary(self, activations: torch.Tensor) -> TopologySummary:
        """Compute topological summary + cheap proxies from activation point cloud."""
        import gudhi
        import numpy as np

        # Deterministic subsampling (stable across calls)
        activations = _deterministic_subsample(activations, self.max_points)

        # Cheap proxies (computed BEFORE PH — always available)
        eff_r = effective_rank(activations)
        cos_c = mean_abs_cosine(activations)

        # Distance matrix
        dm = _activations_to_distance_matrix(activations)
        dm_np = dm.detach().cpu().numpy().astype(np.float64)

        # PH computation
        rips = gudhi.RipsComplex(distance_matrix=dm_np, max_edge_length=float("inf"))
        st = rips.create_simplex_tree(max_dimension=self.max_dimension + 1)
        st.compute_persistence()
        persistence = st.persistence()

        # Extract summary
        betti_0 = 0
        betti_1 = 0
        persistences = []

        for dim, (birth, death) in persistence:
            if death == float("inf"):
                continue
            p = death - birth
            if p > self.persistence_threshold:
                persistences.append(p)
                if dim == 0:
                    betti_0 += 1
                elif dim == 1:
                    betti_1 += 1

        return TopologySummary(
            betti_0=betti_0,
            betti_1=betti_1,
            total_persistence=sum(persistences) if persistences else 0.0,
            max_persistence=max(persistences) if persistences else 0.0,
            mean_persistence=sum(persistences) / len(persistences) if persistences else 0.0,
            num_features=len(persistences),
            eff_rank=eff_r,
            cos_conc=cos_c,
        )

    def set_baseline(self, activations: torch.Tensor):
        """Set the topology baseline from initial (checkpoint) activations.

        This is pi-star in the self-consistency formulation.
        """
        self._baseline = self._compute_summary(activations.detach())
        self._history = [self._baseline]

    def observe(self, activations: torch.Tensor) -> TopologySummary:
        """Observe a new chunk's topology and add to history."""
        summary = self._compute_summary(activations.detach())
        self._history.append(summary)
        if len(self._history) > self.window_size:
            self._history = self._history[-self.window_size :]
        return summary

    def assess(self) -> TopologyState:
        """Assess current topological stability relative to baseline.

        Uses both PH degradation signals AND cheap proxies:
        - Effective rank drop => dimensional collapse
        - Cosine concentration rise => attractor dominance
        - PH feature loss => structural degradation

        Returns:
            TopologyState indicating whether an update is warranted.
        """
        if self._baseline is None or len(self._history) < 2:
            return TopologyState.STABLE

        current = self._history[-1]
        baseline = self._baseline

        # PH degradation score (only penalizes decreases)
        degradation = self._degradation_score(baseline, current)

        # Rate-of-change between consecutive observations
        rate_degradation = 0.0
        if len(self._history) >= 2:
            prev = self._history[-2]
            rate_degradation = self._degradation_score(prev, current)

        # Proxy-based collapse cues
        proxy_score = self._proxy_collapse_score(baseline, current)

        # Absolute divergence: topology has changed significantly from baseline
        # (catches expansion, not just degradation)
        abs_div = self.divergence_from_baseline()

        # Combine: max of degradation signals
        combined = max(degradation, rate_degradation, proxy_score)

        state = TopologyState.STABLE
        if combined >= self.collapse_threshold:
            state = TopologyState.COLLAPSING
        elif combined >= self.drift_threshold or abs_div >= self.divergence_drift_threshold:
            state = TopologyState.DRIFTING

        # EMA baseline update: slowly follow when STABLE
        if self.baseline_mode == "ema" and state == TopologyState.STABLE:
            self._ema_update_baseline(current)

        return state

    @staticmethod
    def _degradation_score(reference: TopologySummary, current: TopologySummary) -> float:
        """Compute topology degradation score (only penalizes decreases).

        Increases in features/persistence (from longer sequences) are
        NOT penalized — only loss of topological structure counts.
        """
        scores = []

        # Betti number decrease (loss of connected components)
        if reference.betti_0 > 0:
            decrease = max(reference.betti_0 - current.betti_0, 0)
            scores.append(decrease / reference.betti_0)
        if reference.betti_1 > 0:
            decrease = max(reference.betti_1 - current.betti_1, 0)
            scores.append(decrease / reference.betti_1)

        # Total persistence decrease (loss of structural features)
        if reference.total_persistence > 0:
            decrease = max(reference.total_persistence - current.total_persistence, 0)
            scores.append(decrease / reference.total_persistence)

        # Feature count decrease
        if reference.num_features > 0:
            decrease = max(reference.num_features - current.num_features, 0)
            scores.append(decrease / reference.num_features)

        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _proxy_collapse_score(baseline: TopologySummary, current: TopologySummary) -> float:
        """Collapse score from cheap proxies (effective rank, cosine concentration).

        These catch dimensional collapse that PH alone may miss.
        """
        scores = []

        # Effective rank drop (lower = more collapsed)
        if baseline.eff_rank > 1.0:
            rank_drop = max(baseline.eff_rank - current.eff_rank, 0) / baseline.eff_rank
            scores.append(rank_drop)

        # Cosine concentration rise (higher = more aligned = collapse)
        if baseline.cos_conc > 0:
            cos_rise = max(current.cos_conc - baseline.cos_conc, 0) / max(1.0 - baseline.cos_conc, 0.01)
            scores.append(min(cos_rise, 1.0))

        return max(scores) if scores else 0.0

    def _ema_update_baseline(self, current: TopologySummary):
        """Slowly update baseline toward current when topology is stable."""
        if self._baseline is None:
            return
        a = self.ema_alpha
        b = self._baseline
        self._baseline = TopologySummary(
            betti_0=int(round((1 - a) * b.betti_0 + a * current.betti_0)),
            betti_1=int(round((1 - a) * b.betti_1 + a * current.betti_1)),
            total_persistence=(1 - a) * b.total_persistence + a * current.total_persistence,
            max_persistence=(1 - a) * b.max_persistence + a * current.max_persistence,
            mean_persistence=(1 - a) * b.mean_persistence + a * current.mean_persistence,
            num_features=int(round((1 - a) * b.num_features + a * current.num_features)),
            eff_rank=(1 - a) * b.eff_rank + a * current.eff_rank,
            cos_conc=(1 - a) * b.cos_conc + a * current.cos_conc,
        )

    def divergence_from_baseline(self) -> float:
        """Scalar divergence measure from baseline (for logging)."""
        if self._baseline is None or len(self._history) < 1:
            return 0.0

        current = self._history[-1]
        baseline = self._baseline

        if baseline.total_persistence > 0:
            return abs(current.total_persistence - baseline.total_persistence) / baseline.total_persistence
        return 0.0

    def reset(self):
        """Clear all tracked state."""
        self._baseline = None
        self._history = []
