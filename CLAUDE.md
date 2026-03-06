# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LiveLoRA is a research project for **live topological LoRA adaptation** — updating LoRA adapter weights during inference using differentiable persistent homology (PH) signals as a topological fidelity loss. No existing work combines topological signals with inference-time LoRA adaptation.

**Status**: Phase 0 validated. Phase 1 strong results (hybrid entropy+PH wins per-prompt). Phase 2 LiveLoRA-Delta validated — PH-Delta wins decisively in chunked generation.

## Git Conventions

- **Always use Gitmoji** syntax for commit messages (e.g., `:sparkles: Add ScaleNet module`, `:bug: Fix gradient flow in PH loss`)
- **Commit frequently** — small, incremental commits serve as progress tracking and context for future coding agents and developers
- Common gitmojis: `:sparkles:` new feature, `:bug:` bugfix, `:recycle:` refactor, `:white_check_mark:` tests, `:memo:` docs, `:fire:` remove code, `:construction:` WIP, `:zap:` performance, `:wrench:` config

## Commands

```bash
# Install (editable) — use py -3 on this Windows machine
py -3 -m pip install -e ".[dev]"

# Run tests (40 tests, ~6s)
py -3 -m pytest tests/ -v

# Run toy experiment (default: Qwen3.5-0.8B, or gpt2 as fallback)
py -3 experiments/toy_ttt.py
py -3 experiments/toy_ttt.py --model gpt2 --steps 5

# Run PH benchmark
py -3 experiments/benchmark_ph.py

# Lint
ruff check livelora/ tests/
```

## Architecture

Two operating modes, built from shared components:

### Per-prompt adaptation (LiveLoRA)
`livelora/core/ttt_loop.py` — N gradient steps on LoRA before generation.

### Chunked generation (LiveLoRA-Delta)
`livelora/core/gen_controller.py` — Generate in 32-token chunks, evaluate topology at each boundary, accept/reject LoRA updates via MDL ratio gate.

### Core modules

1. **`livelora/topology/ph_loss.py`** — Differentiable PH losses using GUDHI. Three modes: persistence maximization, Betti number targeting, diagram divergence. GUDHI wasserstein is optional (requires POT).

2. **`livelora/topology/ph_tracker.py`** — Rolling topology baseline tracker. Computes TopologySummary (Betti numbers, persistence stats) and classifies state as STABLE/DRIFTING/COLLAPSING.

3. **`livelora/core/lora_adapter.py`** — PEFT wrapper with checkpoint/restore, `get_lora_target_modules()` for auto target detection across architectures.

4. **`livelora/core/gen_controller.py`** — LiveLoRA-Delta engine with MDL ratio gate: `rho = delta_struct / (delta_sem + beta)`. Accept iff KL trust region + payoff gate + net improvement. Auto-rollback on reject.

5. **`livelora/core/scalenet.py`** — Per-layer LR modulation (not yet wired into gen_controller).

6. **`livelora/data/chatgpt_loader.py`** — Parses OpenAI ChatGPT export (conversations.json / zip) into (user, assistant) turn pairs.

### Key design decisions
- GUDHI is used for PH (numpy-based), with differentiability through the distance matrix
- **64 points is the sweet spot** for PH (~25-65ms). 128 is borderline (~300ms). 256+ is intractable
- **H0-only is 10x faster than H0+H1** — start with H0 for iteration speed
- Hidden dim barely affects PH time — PCA before PH won't help speed (Rips complex dominates)
- LoRA uses HuggingFace PEFT — `get_lora_target_modules()` auto-detects per architecture
- Adapt-and-reset is default; gen_controller adds KL trust region to prevent semantic drift

## Key Dependencies

- `torch>=2.2`, `transformers>=4.40`, `peft>=0.10` — model + LoRA infrastructure
- `gudhi>=3.9` — persistent homology computation (the topological engine)
- `einops` — tensor reshaping utilities

## Experiments

- **`experiments/toy_ttt.py`** — Proof of concept TTT on GPT-2/Qwen3.5-0.8B
- **`experiments/benchmark_ph.py`** — PH computation speed benchmarks
- **`experiments/correlation_study.py`** — Does topology predict output quality?
- **`experiments/three_way_comparison.py`** — Phase 1: no-adapt vs entropy-TTT vs PH-TTT (per-prompt)
- **`experiments/delta_comparison.py`** — Phase 2: LiveLoRA-Delta chunked generation (PH vs entropy vs hybrid)

## Validated Findings

- TTT loop works end-to-end: loss decreases across steps, LoRA weights change measurably
- Gradients flow through PH → distance matrix → activations → LoRA in 100% of tested cases
- PH computation benchmarked: 64 points at H0+H1 ~25-65ms, scales O(n^3) with point count
- **Correlation study (Qwen3.5-0.8B, GPU, n=20)**: Spearman ρ = -0.39 — moderate negative correlation supports thesis that topology predicts quality
- **4-way comparison (Qwen3.5-0.8B, GPU, n=20)**: Hybrid entropy+PH loss wins decisively (mean 0.307 vs entropy 0.242, wins 14/20 vs entropy-only)
- PH-TTT alone: 13/20 wins vs baseline, mean 0.268
- **PyTorch CUDA**: installed cu124 build. RTX 2060 12GB. bf16 requires float32 cast for cdist, eigvalsh, numpy
- **PHTracker**: directional degradation scoring + absolute divergence threshold (1.5x) for drift trigger
- **Conditional PH escalation**: STABLE=skip, DRIFTING=1 attempt, COLLAPSING=2 attempts
- **LiveLoRA-Delta (Qwen3.5-0.8B, GPU, n=20)**: PH-Delta mean=0.856 beats entropy-Delta 0.776 (10-5 wins) and hybrid-Delta 0.730 (11-4 wins). All beat baseline 0.303. PH's 40% acceptance rate is a feature — stricter gate prevents overfitting.
- **Gen controller optimization modes**: "ph" (structural rho gate), "entropy" (entropy loss, topo gate), "hybrid" (combined loss). PH mode works best in chunked generation; hybrid wins in per-prompt TTT.
