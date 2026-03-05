# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LiveLoRA is a research project for **live topological LoRA adaptation** — updating LoRA adapter weights during inference using differentiable persistent homology (PH) signals as a topological fidelity loss. No existing work combines topological signals with inference-time LoRA adaptation.

**Status**: Phase 0 — core loop implemented, awaiting validation experiments.

## Git Conventions

- **Always use Gitmoji** syntax for commit messages (e.g., `:sparkles: Add ScaleNet module`, `:bug: Fix gradient flow in PH loss`)
- **Commit frequently** — small, incremental commits serve as progress tracking and context for future coding agents and developers
- Common gitmojis: `:sparkles:` new feature, `:bug:` bugfix, `:recycle:` refactor, `:white_check_mark:` tests, `:memo:` docs, `:fire:` remove code, `:construction:` WIP, `:zap:` performance, `:wrench:` config

## Commands

```bash
# Install (editable)
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run toy experiment (default: Qwen3.5-0.8B, or gpt2 as fallback)
python experiments/toy_ttt.py
python experiments/toy_ttt.py --model gpt2 --steps 5  # smaller download

# Lint
ruff check livelora/ tests/
```

## Architecture

The system has three layers, built bottom-up:

1. **`livelora/topology/ph_loss.py`** — Differentiable PH loss functions using GUDHI. Computes persistent homology on activation distance matrices. Three loss modes: persistence maximization, Betti number targeting, diagram divergence. Gradients flow through the distance matrix back to activations.

2. **`livelora/core/lora_adapter.py`** — PEFT wrapper (`LiveLoraModel`) with checkpoint/restore for adapt-and-reset pattern, selective gradient control (only LoRA params trainable), and L2 drift regularization from checkpoint.

3. **`livelora/core/ttt_loop.py`** — The test-time training engine (`TTTLoop`). Orchestrates: forward pass → extract activations → PH loss → backprop → LoRA update. Configurable steps, layers, subsample size.

4. **`livelora/core/scalenet.py`** — Per-layer learning rate modulation (not yet integrated into TTT loop). Predicts LR scale factors from topological signal features.

### Key design decisions
- GUDHI is used for PH (numpy-based), with careful handling to maintain differentiability through the distance matrix
- PH is computed on subsampled activation point clouds (default 64-256 points) for tractability
- LoRA uses HuggingFace PEFT — `get_lora_target_modules()` auto-detects target layers per architecture (q_proj/v_proj for Qwen/Llama/Gemma, c_attn for GPT-2)
- Adapt-and-reset is the default: checkpoint LoRA before each query, restore after generation

## Key Dependencies

- `torch>=2.2`, `transformers>=4.40`, `peft>=0.10` — model + LoRA infrastructure
- `gudhi>=3.9` — persistent homology computation (the topological engine)
- `einops` — tensor reshaping utilities

## Known Limitations / Open Questions

- GUDHI detaches to numpy for PH computation — gradients flow through the distance matrix indexing, but this may cause sparse/noisy gradients
- PH computation scales poorly beyond ~256 points — subsample aggressively
- It's unproven whether topological loss actually improves LLM outputs — Phase 1 experiments will answer this
