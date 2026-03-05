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
    """Compact summary of a persistence diagram for tracking."""

    betti_0: int = 0  # Number of connected components above threshold
    betti_1: int = 0  # Number of loops above threshold
    total_persistence: float = 0.0  # Sum of all (death - birth) values
    max_persistence: float = 0.0  # Largest single feature persistence
    mean_persistence: float = 0.0  # Average persistence
    num_features: int = 0  # Total features above threshold


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
    ):
        """
        Args:
            max_points: Subsample activation points to this count.
            max_dimension: Max homological dimension for PH.
            persistence_threshold: Min persistence to count a feature.
            drift_threshold: Relative change from baseline to flag drifting.
            collapse_threshold: Relative change from baseline to flag collapse.
            window_size: Number of recent summaries to keep for rolling stats.
        """
        self.max_points = max_points
        self.max_dimension = max_dimension
        self.persistence_threshold = persistence_threshold
        self.drift_threshold = drift_threshold
        self.collapse_threshold = collapse_threshold
        self.window_size = window_size

        self._baseline: TopologySummary | None = None
        self._history: list[TopologySummary] = []
        self._ph_loss = DifferentiablePHLoss(
            max_dimension=max_dimension,
            max_points=max_points,
            persistence_threshold=persistence_threshold,
        )

    def _compute_summary(self, activations: torch.Tensor) -> TopologySummary:
        """Compute topological summary from activation point cloud."""
        import gudhi
        import numpy as np

        # Subsample
        if activations.shape[0] > self.max_points:
            indices = torch.randperm(activations.shape[0])[: self.max_points]
            activations = activations[indices]

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

        Distinguishes topology *degradation* (features disappearing,
        persistence dropping) from topology *expansion* (more features
        from growing sequences, which is normal).

        Only degradation triggers DRIFTING/COLLAPSING states.

        Returns:
            TopologyState indicating whether an update is warranted.
        """
        if self._baseline is None or len(self._history) < 2:
            return TopologyState.STABLE

        current = self._history[-1]
        baseline = self._baseline

        # Compute directional degradation score (only penalize decreases)
        degradation = self._degradation_score(baseline, current)

        # Also check rate-of-change between consecutive observations
        rate_degradation = 0.0
        if len(self._history) >= 2:
            prev = self._history[-2]
            rate_degradation = self._degradation_score(prev, current)

        # Combine: sustained degradation from baseline + sudden drops
        combined = max(degradation, rate_degradation)

        if combined >= self.collapse_threshold:
            return TopologyState.COLLAPSING
        elif combined >= self.drift_threshold:
            return TopologyState.DRIFTING
        else:
            return TopologyState.STABLE

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
