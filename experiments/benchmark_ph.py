"""Benchmark PH computation time vs. activation size.

Sweeps max_points from 16 to 512 and measures:
- PH computation time (GUDHI)
- Number of persistence features found
- Gradient computation time (backward through distance matrix)

Run:
    python experiments/benchmark_ph.py
"""

from __future__ import annotations

import time

import torch

from livelora.topology.ph_loss import DifferentiablePHLoss


def benchmark_one(n_points: int, hidden_dim: int, max_dim: int = 1, n_trials: int = 5):
    """Benchmark PH computation for a given point cloud size."""
    loss_fn = DifferentiablePHLoss(
        max_dimension=max_dim,
        max_points=n_points,
        target_betti={0: 1, 1: 0},
    )

    fwd_times = []
    bwd_times = []
    n_features = []

    for _ in range(n_trials):
        points = torch.randn(n_points, hidden_dim, requires_grad=True)

        # Forward (PH computation)
        t0 = time.perf_counter()
        loss = loss_fn(points)
        t1 = time.perf_counter()
        fwd_times.append(t1 - t0)

        # Backward (gradient through distance matrix)
        t2 = time.perf_counter()
        loss.backward()
        t3 = time.perf_counter()
        bwd_times.append(t3 - t2)

        has_grad = points.grad is not None and points.grad.abs().sum().item() > 0
        n_features.append(1 if has_grad else 0)  # crude proxy

    return {
        "n_points": n_points,
        "hidden_dim": hidden_dim,
        "max_dim": max_dim,
        "fwd_ms": 1000 * sum(fwd_times) / len(fwd_times),
        "bwd_ms": 1000 * sum(bwd_times) / len(bwd_times),
        "total_ms": 1000 * (sum(fwd_times) + sum(bwd_times)) / len(fwd_times),
        "grad_nonzero": sum(n_features) / len(n_features),
    }


def main():
    print("=" * 75)
    print("PH Computation Benchmark")
    print("=" * 75)

    # Sweep point counts
    point_counts = [16, 32, 64, 128, 256, 512]
    hidden_dims = [64, 256, 1024]

    print(f"\n{'points':>8} {'h_dim':>6} {'max_d':>6} {'fwd_ms':>10} {'bwd_ms':>10} {'total_ms':>10} {'grad_ok':>8}")
    print("-" * 75)

    for h_dim in hidden_dims:
        for n_points in point_counts:
            result = benchmark_one(n_points, h_dim, max_dim=1, n_trials=3)
            print(
                f"{result['n_points']:>8} "
                f"{result['hidden_dim']:>6} "
                f"{result['max_dim']:>6} "
                f"{result['fwd_ms']:>10.2f} "
                f"{result['bwd_ms']:>10.2f} "
                f"{result['total_ms']:>10.2f} "
                f"{result['grad_nonzero']:>8.0%}"
            )

    # Compare H0-only vs H0+H1
    print(f"\n\nH0 only vs H0+H1 (128 points, 256 dims)")
    print("-" * 50)
    for max_dim in [0, 1]:
        result = benchmark_one(128, 256, max_dim=max_dim, n_trials=5)
        label = f"H0..H{max_dim}"
        print(f"  {label}: fwd={result['fwd_ms']:.1f}ms  bwd={result['bwd_ms']:.1f}ms  total={result['total_ms']:.1f}ms")

    # PCA test: does reducing hidden_dim help?
    print(f"\n\nPCA effect: 128 points, varying hidden dim")
    print("-" * 50)
    for h_dim in [8, 16, 32, 64, 128, 256, 512, 1024]:
        result = benchmark_one(128, h_dim, max_dim=1, n_trials=3)
        print(f"  dim={h_dim:>5}: fwd={result['fwd_ms']:.1f}ms  bwd={result['bwd_ms']:.1f}ms  total={result['total_ms']:.1f}ms")


if __name__ == "__main__":
    main()
