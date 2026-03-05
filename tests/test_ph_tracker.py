"""Tests for PH topology tracker."""

import pytest
import torch

gudhi = pytest.importorskip("gudhi")

from livelora.topology.ph_tracker import PHTracker, TopologyState, TopologySummary


class TestTopologySummary:
    def test_defaults(self):
        s = TopologySummary()
        assert s.betti_0 == 0
        assert s.total_persistence == 0.0


class TestPHTracker:
    @pytest.fixture
    def tracker(self):
        return PHTracker(max_points=32, max_dimension=0)

    def test_set_baseline(self, tracker):
        points = torch.randn(32, 16)
        tracker.set_baseline(points)
        assert tracker._baseline is not None
        assert len(tracker._history) == 1

    def test_observe_adds_to_history(self, tracker):
        points = torch.randn(32, 16)
        tracker.set_baseline(points)
        tracker.observe(torch.randn(32, 16))
        tracker.observe(torch.randn(32, 16))
        assert len(tracker._history) == 3

    def test_stable_when_similar(self, tracker):
        # Same distribution should be stable
        points = torch.randn(32, 16)
        tracker.set_baseline(points)
        tracker.observe(points + 0.01 * torch.randn_like(points))
        state = tracker.assess()
        assert state == TopologyState.STABLE

    def test_detects_collapse(self, tracker):
        # Normal data as baseline
        tracker.set_baseline(torch.randn(32, 16))
        # First observation: slight drift
        tracker.observe(torch.randn(32, 16) * 0.5)
        # Second observation: sudden collapse (all same point)
        tracker.observe(torch.ones(32, 16) + 0.001 * torch.randn(32, 16))
        state = tracker.assess()
        # Should detect significant change (either drifting or collapsing)
        assert state in (TopologyState.DRIFTING, TopologyState.COLLAPSING)

    def test_reset_clears_state(self, tracker):
        tracker.set_baseline(torch.randn(32, 16))
        tracker.reset()
        assert tracker._baseline is None
        assert len(tracker._history) == 0

    def test_window_size_respected(self):
        tracker = PHTracker(max_points=16, max_dimension=0, window_size=3)
        tracker.set_baseline(torch.randn(16, 8))
        for _ in range(10):
            tracker.observe(torch.randn(16, 8))
        assert len(tracker._history) <= 3

    def test_divergence_from_baseline(self, tracker):
        tracker.set_baseline(torch.randn(32, 16))
        tracker.observe(torch.randn(32, 16))
        div = tracker.divergence_from_baseline()
        assert isinstance(div, float)
        assert div >= 0.0
