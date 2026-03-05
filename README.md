# LiveLoRA

**Inference-time LoRA refinement via differentiable persistent homology.**

LiveLoRA updates LoRA adapter weights *during inference* using topological fidelity signals. Instead of generating a static adapter and freezing it, LiveLoRA runs a short test-time training (TTT) loop that computes differentiable persistent homology on model activations and backpropagates through LoRA parameters to improve representational quality per-instance.

This is a novel intersection — no existing work combines topological signals with inference-time LoRA adaptation.

## How It Works

```
Input → Forward pass with LoRA → Extract activations → Compute PH loss → Backprop → Update LoRA → Repeat (1-3 steps)
```

1. **Warm-start** (future): A hypernetwork produces an initial LoRA adapter from the input in one forward pass
2. **Topological refinement**: For each input, compute persistent homology on activation point clouds, define a topological fidelity loss, and update LoRA weights with a few gradient steps
3. **Stability**: Adapt-and-reset pattern (checkpoint/restore LoRA per query) with L2 drift regularization and gradient clipping

### What is persistent homology doing here?

Persistent homology captures the *shape* of data — connected components, loops, voids — across scales. Applied to model activations, it tells us whether the internal representations have the right topological structure. The PH loss is fully differentiable, so gradients flow from topological features through the distance matrix back to LoRA parameters.

## Architecture

```
livelora/
├── core/
│   ├── lora_adapter.py    # PEFT wrapper with checkpoint/restore, selective gradients
│   ├── ttt_loop.py        # Test-time training engine (the main loop)
│   └── scalenet.py        # Per-layer LR modulation (surprise-proportional)
├── topology/
│   └── ph_loss.py         # Differentiable PH losses (persistence, Betti, divergence)
└── warmstart/             # (Planned) Perceiver-style initialization hypernetwork
```

## Quick Start

### Install

```bash
# Base install
pip install -e .

# With dev tools
pip install -e ".[dev]"

# With experiment tracking
pip install -e ".[experiment]"

# Everything
pip install -e ".[full]"
```

**Requirements**: Python 3.10+, PyTorch 2.2+, GUDHI 3.9+, HuggingFace PEFT & Transformers.

### Run the toy experiment

Validates the full loop end-to-end on GPT-2 (CPU-compatible):

```bash
python experiments/toy_ttt.py --model gpt2 --steps 5 --rank 4
```

This will:
- Load GPT-2 with a rank-4 LoRA
- Run 5 TTT steps with PH-based topological loss
- Report loss trajectory and LoRA weight changes

### Run tests

```bash
pytest tests/
```

## Key Concepts

| Concept | What it means in LiveLoRA |
|---|---|
| **Test-time training (TTT)** | Updating model weights during inference, not just during training |
| **Persistent homology (PH)** | Mathematical tool that captures topological features (components, loops) of point clouds at multiple scales |
| **Topological fidelity loss** | A differentiable loss that penalizes when activations have the wrong topological structure |
| **Adapt-and-reset** | Checkpoint LoRA before each input, refine, generate, then restore — each query gets fresh adaptation |
| **ScaleNet** | A tiny network that predicts per-layer learning rates based on the current topological signal (surprise-proportional) |

## Research Foundations

This project synthesizes ideas from:

- **LoRA-TTT** (arXiv:2502.02069) — Test-time LoRA adaptation with entropy loss
- **Unsupervised Dynamic TTA** (arXiv:2602.09719) — Per-layer LR modulation via ScaleNet
- **Titans** (arXiv:2501.00663) — Surprise-proportional weight updates
- **Clough et al.** (IEEE TPAMI 2020) — Differentiable PH losses for neural networks
- **Text-to-LoRA / Doc-to-LoRA** (Sakana AI) — Hypernetwork-based adapter generation (inspiration for warm-start)

See [`LiveLoRA - Brief.md`](LiveLoRA%20-%20Brief.md) for the full research analysis.

## Current Status

**Phase 0 — Foundation.** The core loop is implemented and ready for validation. See [ROADMAP.md](ROADMAP.md) for the full plan.

## License

Apache 2.0
