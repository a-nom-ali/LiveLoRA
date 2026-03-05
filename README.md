# LiveLoRA

**Inference-time LoRA refinement via differentiable persistent homology.**

LiveLoRA updates LoRA adapter weights *during inference* using topological fidelity signals. Instead of generating a static adapter and freezing it, LiveLoRA runs a short test-time training (TTT) loop that computes differentiable persistent homology on model activations and backpropagates through LoRA parameters to improve representational quality per-instance.

This is a novel intersection — no existing work combines topological signals with inference-time LoRA adaptation.

## How It Works

Two modes of operation:

### Per-prompt adaptation (LiveLoRA)

```
Input → Forward pass with LoRA → Extract activations → Compute PH loss → Backprop → Update LoRA → Repeat (1-3 steps) → Generate
```

### Chunked generation adaptation (LiveLoRA-Delta)

```
Prompt → Generate chunk (32 tokens) → Check topology → Update LoRA if destabilized → Generate next chunk → ...
```

LiveLoRA-Delta adapts *during* generation at chunk boundaries, using topology as a "structural health check." A gating mechanism only triggers LoRA updates when the activation manifold destabilizes — making it "live" without constant training overhead.

### Core components

1. **Topological refinement**: Compute persistent homology on activation point clouds, define a topological fidelity loss, and update LoRA weights with a few gradient steps
2. **Topology gating**: Only adapt when needed — stable topology skips updates, destabilizing topology triggers small corrections, collapsing topology triggers stronger intervention
3. **Stability**: Adapt-and-reset pattern (checkpoint/restore LoRA per query) with L2 drift regularization, gradient clipping, and update cooldowns
4. **Warm-start** (future): A hypernetwork produces an initial LoRA adapter from the input in one forward pass

### What is persistent homology doing here?

Persistent homology captures the *shape* of data — connected components, loops, voids — across scales. Applied to model activations, it tells us whether the internal representations have the right topological structure. The PH loss is fully differentiable, so gradients flow from topological features through the distance matrix back to LoRA parameters.

Most representation learning optimizes distance, similarity, entropy, or contrast. LiveLoRA optimizes **topological invariants** — a qualitatively different signal that captures the geometry of the representation manifold itself.

## Architecture

```
livelora/
├── core/
│   ├── lora_adapter.py    # PEFT wrapper with checkpoint/restore, auto target detection
│   ├── ttt_loop.py        # Test-time training engine (per-prompt adaptation)
│   ├── gen_controller.py  # (Planned) Chunked generation with topology gating
│   └── scalenet.py        # Per-layer LR modulation (surprise-proportional)
├── topology/
│   ├── ph_loss.py         # Differentiable PH losses (persistence, Betti, divergence)
│   └── ph_tracker.py      # (Planned) Rolling topology baseline + divergence detection
├── data/
│   └── chatgpt_loader.py  # OpenAI ChatGPT export parser for test data
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

Validates the full loop end-to-end (default: Qwen3.5-0.8B):

```bash
# Default: Qwen3.5-0.8B (~3GB VRAM with LoRA, runs on CPU too)
python experiments/toy_ttt.py

# Larger model for better quality
python experiments/toy_ttt.py --model Qwen/Qwen3.5-2B --steps 3

# GPT-2 fallback (smallest download, good for CI)
python experiments/toy_ttt.py --model gpt2 --steps 5
```

This will:
- Load the model with a rank-4 LoRA (auto-detects target modules)
- Run TTT steps with PH-based topological loss
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
| **LiveLoRA-Delta** | Chunked generation with topology-gated adaptation — adapt mid-generation only when the manifold destabilizes |
| **Adapt-and-reset** | Checkpoint LoRA before each input, refine, generate, then restore — each query gets fresh adaptation |
| **ScaleNet** | A tiny network that predicts per-layer learning rates based on the current topological signal (surprise-proportional) |
| **Topology gating** | Only update LoRA when PH loss exceeds threshold or deviates from rolling baseline — skip updates when topology is stable |

## Research Foundations

This project synthesizes ideas from:

- **LoRA-TTT** (arXiv:2502.02069) — Test-time LoRA adaptation with entropy loss
- **Unsupervised Dynamic TTA** (arXiv:2602.09719) — Per-layer LR modulation via ScaleNet
- **Titans** (arXiv:2501.00663) — Surprise-proportional weight updates
- **Clough et al.** (IEEE TPAMI 2020) — Differentiable PH losses for neural networks
- **Text-to-LoRA / Doc-to-LoRA** (Sakana AI) — Hypernetwork-based adapter generation (inspiration for warm-start)

See [`LiveLoRA - Brief.md`](LiveLoRA%20-%20Brief.md) for the full research analysis and [`ChatGPT - Feedback.md`](ChatGPT%20-%20Feedback.md) for detailed technical feedback that shaped the roadmap.

## Current Status

**Phase 0 — Foundation.** The core loop is implemented and ready for validation. See [ROADMAP.md](ROADMAP.md) for the full plan.

## License

Apache 2.0
