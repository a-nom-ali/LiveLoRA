"""Differentiable persistent homology losses for representation fidelity.

Uses GUDHI for PH computation with PyTorch autograd integration.
The key idea: compute persistence diagrams from model activations,
define a loss that penalizes topological infidelity, and backprop
through LoRA parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

try:
    import gudhi
    from gudhi.wasserstein import wasserstein_distance as gudhi_wasserstein

    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False


@dataclass
class PersistenceDiagram:
    """A persistence diagram: paired (birth, death) values plus dimension info."""

    pairs: torch.Tensor  # shape (n_features, 2) — birth/death
    dimensions: torch.Tensor  # shape (n_features,) — homological dimension


def _activations_to_distance_matrix(activations: torch.Tensor) -> torch.Tensor:
    """Convert activation vectors to a pairwise distance matrix.

    Args:
        activations: (n_points, d) tensor of activation vectors
            (e.g., token embeddings from a specific layer).

    Returns:
        (n_points, n_points) pairwise Euclidean distance matrix.
    """
    # Use cdist for numerical stability
    return torch.cdist(activations.unsqueeze(0), activations.unsqueeze(0)).squeeze(0)


class DifferentiablePHLoss(nn.Module):
    """Compute a differentiable topological loss from model activations.

    Strategy: compute PH on the activation distance matrix, identify
    persistent features, and define a loss that penalizes deviations
    from a target topological signature (e.g., target Betti numbers
    or a reference persistence diagram).

    The gradient flows through the distance matrix back to the activations,
    and from there through the model to the LoRA parameters.
    """

    def __init__(
        self,
        max_dimension: int = 1,
        max_points: int = 256,
        target_betti: dict[int, int] | None = None,
        persistence_threshold: float = 0.01,
    ):
        """
        Args:
            max_dimension: Maximum homological dimension to compute (0 = components, 1 = loops).
            max_points: Subsample activations to this many points for tractable PH.
            target_betti: Target Betti numbers per dimension, e.g. {0: 1, 1: 0}.
                If None, uses a persistence-maximization objective.
            persistence_threshold: Minimum persistence to count a feature as real (not noise).
        """
        super().__init__()
        if not GUDHI_AVAILABLE:
            raise ImportError(
                "GUDHI is required for PH computation. Install with: pip install gudhi"
            )
        self.max_dimension = max_dimension
        self.max_points = max_points
        self.target_betti = target_betti
        self.persistence_threshold = persistence_threshold

    def _subsample(self, activations: torch.Tensor) -> torch.Tensor:
        """Subsample activation points if needed for tractable PH."""
        n = activations.shape[0]
        if n <= self.max_points:
            return activations
        indices = torch.randperm(n, device=activations.device)[: self.max_points]
        return activations[indices]

    def _compute_persistence(
        self, distance_matrix: torch.Tensor
    ) -> list[tuple[int, tuple[float, float]]]:
        """Compute persistence diagram using GUDHI on a distance matrix.

        This detaches to numpy for GUDHI, but we use the birth/death values
        to index back into the differentiable distance matrix.
        """
        dm_np = distance_matrix.detach().cpu().numpy().astype(np.float64)
        rips = gudhi.RipsComplex(distance_matrix=dm_np, max_edge_length=float("inf"))
        st = rips.create_simplex_tree(max_dimension=self.max_dimension + 1)
        st.compute_persistence()
        return st.persistence_pairs()

    def forward(
        self,
        activations: torch.Tensor,
        reference_activations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute topological fidelity loss.

        Args:
            activations: (n_points, d) activation tensor from the LoRA-adapted model.
            reference_activations: Optional (n_points, d) activations from the base model
                (without LoRA). If provided, loss penalizes topological divergence.

        Returns:
            Scalar loss tensor with gradients flowing to activations.
        """
        activations = self._subsample(activations)
        dist_matrix = _activations_to_distance_matrix(activations)

        if reference_activations is not None:
            return self._divergence_loss(activations, reference_activations, dist_matrix)
        elif self.target_betti is not None:
            return self._betti_loss(dist_matrix)
        else:
            return self._persistence_loss(dist_matrix)

    def _persistence_loss(self, dist_matrix: torch.Tensor) -> torch.Tensor:
        """Maximize total persistence — encourage well-separated topological features."""
        pairs = self._compute_persistence(dist_matrix)
        total_persistence = torch.tensor(0.0, device=dist_matrix.device)

        for simplex_birth, simplex_death in pairs:
            if len(simplex_birth) == 0 or len(simplex_death) == 0:
                continue
            # Birth value = max edge weight in birth simplex
            # Death value = max edge weight in death simplex
            birth_val = self._simplex_filtration_value(simplex_birth, dist_matrix)
            death_val = self._simplex_filtration_value(simplex_death, dist_matrix)
            persistence = death_val - birth_val
            if persistence > self.persistence_threshold:
                total_persistence = total_persistence + persistence

        # Negate: we want to maximize persistence
        return -total_persistence

    def _betti_loss(self, dist_matrix: torch.Tensor) -> torch.Tensor:
        """Penalize deviation from target Betti numbers at a fixed scale."""
        pairs = self._compute_persistence(dist_matrix)
        loss = torch.tensor(0.0, device=dist_matrix.device)

        for dim, target_count in self.target_betti.items():
            # Count features in this dimension above persistence threshold
            dim_persistences = []
            for simplex_birth, simplex_death in pairs:
                # Determine homological dimension from simplex sizes
                h_dim = len(simplex_birth) - 1
                if h_dim != dim or len(simplex_death) == 0:
                    continue
                birth_val = self._simplex_filtration_value(simplex_birth, dist_matrix)
                death_val = self._simplex_filtration_value(simplex_death, dist_matrix)
                persistence = death_val - birth_val
                if persistence > self.persistence_threshold:
                    dim_persistences.append(persistence)

            actual_count = len(dim_persistences)
            # Soft penalty for count mismatch
            count_diff = actual_count - target_count
            loss = loss + count_diff**2

            # If we have the right count, also encourage high persistence
            if dim_persistences:
                total_p = sum(dim_persistences)
                loss = loss - 0.1 * total_p

        return loss

    def _divergence_loss(
        self,
        activations: torch.Tensor,
        reference_activations: torch.Tensor,
        dist_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize topological divergence from reference (base model) activations.

        Uses Wasserstein distance between persistence diagrams as a
        differentiable-friendly proxy.
        """
        ref_activations = self._subsample(reference_activations)
        ref_dist_matrix = _activations_to_distance_matrix(ref_activations)

        # Compute persistence diagrams for both
        pairs_adapted = self._compute_persistence(dist_matrix)
        pairs_reference = self._compute_persistence(ref_dist_matrix)

        # Extract birth-death pairs per dimension and compute Wasserstein distance
        loss = torch.tensor(0.0, device=dist_matrix.device)

        for dim in range(self.max_dimension + 1):
            dgm_adapted = self._extract_diagram(pairs_adapted, dim, dist_matrix)
            dgm_reference = self._extract_diagram(pairs_reference, dim, ref_dist_matrix)

            if dgm_adapted.shape[0] == 0 and dgm_reference.shape[0] == 0:
                continue

            # Use L2 distance between sorted persistence values as a differentiable proxy
            pers_adapted = (dgm_adapted[:, 1] - dgm_adapted[:, 0]).sort(descending=True).values
            pers_reference = (
                (dgm_reference[:, 1] - dgm_reference[:, 0]).sort(descending=True).values
            )

            # Pad shorter diagram
            max_len = max(pers_adapted.shape[0], pers_reference.shape[0])
            if max_len == 0:
                continue
            pa = torch.zeros(max_len, device=dist_matrix.device)
            pr = torch.zeros(max_len, device=dist_matrix.device)
            pa[: pers_adapted.shape[0]] = pers_adapted
            pr[: pers_reference.shape[0]] = pers_reference

            loss = loss + torch.sum((pa - pr) ** 2)

        return loss

    def _extract_diagram(
        self,
        pairs: list[tuple],
        dim: int,
        dist_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Extract (birth, death) tensor for a specific homological dimension."""
        points = []
        for simplex_birth, simplex_death in pairs:
            h_dim = len(simplex_birth) - 1
            if h_dim != dim or len(simplex_death) == 0:
                continue
            birth_val = self._simplex_filtration_value(simplex_birth, dist_matrix)
            death_val = self._simplex_filtration_value(simplex_death, dist_matrix)
            if (death_val - birth_val) > self.persistence_threshold:
                points.append(torch.stack([birth_val, death_val]))

        if not points:
            return torch.zeros(0, 2, device=dist_matrix.device)
        return torch.stack(points)

    @staticmethod
    def _simplex_filtration_value(simplex: tuple, dist_matrix: torch.Tensor) -> torch.Tensor:
        """Get the filtration value of a simplex (max pairwise distance among its vertices).

        This remains differentiable because we index into the differentiable distance matrix.
        """
        vertices = list(simplex)
        if len(vertices) == 1:
            return torch.tensor(0.0, device=dist_matrix.device)
        max_val = dist_matrix[vertices[0], vertices[1]]
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                edge = dist_matrix[vertices[i], vertices[j]]
                max_val = torch.maximum(max_val, edge)
        return max_val
