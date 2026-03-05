"""Tests for differentiable PH loss."""

import pytest
import torch

gudhi = pytest.importorskip("gudhi")

from livelora.topology.ph_loss import DifferentiablePHLoss, _activations_to_distance_matrix


class TestDistanceMatrix:
    def test_shape(self):
        points = torch.randn(10, 32)
        dm = _activations_to_distance_matrix(points)
        assert dm.shape == (10, 10)

    def test_symmetric(self):
        points = torch.randn(10, 32)
        dm = _activations_to_distance_matrix(points)
        assert torch.allclose(dm, dm.T, atol=1e-6)

    def test_zero_diagonal(self):
        points = torch.randn(10, 32)
        dm = _activations_to_distance_matrix(points)
        assert torch.allclose(dm.diag(), torch.zeros(10), atol=1e-6)

    def test_differentiable(self):
        points = torch.randn(10, 32, requires_grad=True)
        dm = _activations_to_distance_matrix(points)
        dm.sum().backward()
        assert points.grad is not None


class TestPHLoss:
    def test_persistence_loss_runs(self):
        loss_fn = DifferentiablePHLoss(max_dimension=0, max_points=20)
        points = torch.randn(20, 8, requires_grad=True)
        loss = loss_fn(points)
        assert loss.shape == ()
        loss.backward()
        assert points.grad is not None

    def test_betti_loss_runs(self):
        loss_fn = DifferentiablePHLoss(
            max_dimension=1,
            max_points=20,
            target_betti={0: 1, 1: 0},
        )
        points = torch.randn(20, 8, requires_grad=True)
        loss = loss_fn(points)
        assert loss.shape == ()

    def test_divergence_loss_runs(self):
        loss_fn = DifferentiablePHLoss(max_dimension=0, max_points=20)
        adapted = torch.randn(20, 8, requires_grad=True)
        reference = torch.randn(20, 8)
        loss = loss_fn(adapted, reference_activations=reference)
        assert loss.shape == ()
        loss.backward()
        assert adapted.grad is not None

    def test_subsample(self):
        loss_fn = DifferentiablePHLoss(max_dimension=0, max_points=10)
        big = torch.randn(100, 8)
        small = loss_fn._subsample(big)
        assert small.shape[0] == 10
