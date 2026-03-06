# LiveLoRA

**Inference-time LoRA refinement via differentiable persistent homology.**

LiveLoRA updates LoRA adapter weights *during inference* using topological fidelity signals. Instead of generating a static adapter and freezing it, LiveLoRA runs topology-gated adaptation that computes differentiable persistent homology on model activations and only applies LoRA updates when they pass a structural quality filter.

The emerging insight: **topology works best as a control law for adaptive inference** — deciding *when* and *whether* to adapt, not just *how*.

## Results

Tested on Qwen3.5-0.8B with 20 prompts, GPU (RTX 2060 12GB):

### LiveLoRA-Delta (chunked generation)

| Method | Consistency | vs Baseline | Acceptance Rate |
|---|---|---|---|
| **PH-Delta** | **0.874** | **20/20 wins** | 39% |
| Entropy-Delta | 0.815 | 20/20 wins | 100% |
| Hybrid-Delta | 0.796 | 20/20 wins | 100% |
| Baseline | 0.255 | — | — |

PH-Delta's 39% acceptance rate is the key: by rejecting ~60% of candidate updates, the structural gate prevents entropy overfitting and keeps only genuinely beneficial changes.

### Per-prompt TTT

| Method | Consistency | vs Baseline |
|---|---|---|
| **Hybrid (entropy+PH)** | **0.307** | **14/20 vs entropy** |
| PH-TTT | 0.268 | 13/20 |
| Entropy-TTT | 0.242 | 10/20 |
| Baseline | 0.230 | — |

### Topology-quality correlation

Spearman ρ = -0.39 between topology divergence and self-consistency — moderate correlation supports the thesis that activation topology predicts output quality.

## How It Works

Two modes of operation:

### Per-prompt adaptation (LiveLoRA)

```
Input → Forward pass with LoRA → Extract activations → Compute PH loss → Backprop → Update LoRA → Generate
```

### Chunked generation (LiveLoRA-Delta)

```
Prompt → Generate chunk → Observe topology → STABLE? skip → DRIFTING? try update → Accept/reject via structural gate → Next chunk
```

LiveLoRA-Delta adapts *during* generation at chunk boundaries. A topology tracker monitors the activation manifold and only triggers LoRA updates when topology diverges from the baseline. The MDL ratio gate then evaluates each candidate update:

1. **KL trust region**: semantic drift must be small
2. **Structural gate**: topology must improve (rho = structural gain / semantic cost)
3. **Net improvement**: the update must make things measurably better

### What persistent homology does here

PH captures the *shape* of activation point clouds — connected components, loops, voids — across scales. Applied to model activations, it provides a global structural coherence signal that's qualitatively different from entropy or distance-based losses. The PH loss is differentiable through the distance matrix, so gradients flow back to LoRA parameters.

## Architecture

```
livelora/
├── core/
│   ├── lora_adapter.py    # PEFT wrapper with checkpoint/restore, auto target detection
│   ├── ttt_loop.py        # Test-time training engine (per-prompt adaptation)
│   ├── gen_controller.py  # LiveLoRA-Delta: chunked generation with topology gating
│   └── scalenet.py        # Per-layer LR modulation
├── topology/
│   ├── ph_loss.py         # Differentiable PH losses (persistence, Betti, divergence)
│   ├── ph_tracker.py      # Rolling topology tracker (STABLE/DRIFTING/COLLAPSING)
│   └── entropy_loss.py    # Entropy-based TTT loss (baseline competitor)
├── data/
│   └── chatgpt_loader.py  # OpenAI ChatGPT export parser for test data
└── warmstart/             # (Planned) Perceiver-style initialization hypernetwork
```

## Quick Start

### Install

```bash
pip install -e ".[dev]"
```

**Requirements**: Python 3.10+, PyTorch 2.2+, GUDHI 3.9+, HuggingFace PEFT & Transformers.

### Run experiments

```bash
# Toy TTT validation
python experiments/toy_ttt.py

# Correlation study: does topology predict quality?
python experiments/correlation_study.py --model Qwen/Qwen3.5-0.8B

# Per-prompt comparison (4 methods)
python experiments/three_way_comparison.py --model Qwen/Qwen3.5-0.8B

# LiveLoRA-Delta comparison (chunked generation, 4 methods)
python experiments/delta_comparison.py --model Qwen/Qwen3.5-0.8B --device auto

# Gate ablation: isolate where the win comes from
python experiments/gate_ablation.py --model Qwen/Qwen3.5-0.8B --device auto
```

### Run tests

```bash
pytest tests/  # 40 tests, ~6s
```

## Key Concepts

| Concept | What it means in LiveLoRA |
|---|---|
| **Test-time training (TTT)** | Updating model weights during inference |
| **Persistent homology (PH)** | Captures topological features of point clouds at multiple scales |
| **LiveLoRA-Delta** | Chunked generation with topology-gated adaptation |
| **Structural gate** | Accept LoRA updates only if topology improves efficiently |
| **KL trust region** | Pin output distribution close to pre-update state |
| **MDL ratio (rho)** | Structural improvement / semantic displacement — high rho = efficient repair |
| **Divergence drift threshold** | Trigger adaptation when topology diverges 1.5x from baseline |

## Research Foundations

- **LoRA-TTT** (arXiv:2502.02069) — Test-time LoRA adaptation with entropy loss
- **Unsupervised Dynamic TTA** (arXiv:2602.09719) — Per-layer LR modulation via ScaleNet
- **Titans** (arXiv:2501.00663) — Surprise-proportional weight updates
- **Clough et al.** (IEEE TPAMI 2020) — Differentiable PH losses for neural networks

See [`LiveLoRA - Brief.md`](LiveLoRA%20-%20Brief.md) for the full research analysis and [`ROADMAP.md`](ROADMAP.md) for the development plan.

## Current Status

**Phase 2 validated.** PH-Delta wins decisively in chunked generation. Gate ablation study in progress to isolate the mechanism. See [ROADMAP.md](ROADMAP.md) for details.

## License

Apache 2.0
