# LiveLoRA Roadmap

This roadmap is organized into phases, each with clear goals and deliverables. Each phase answers a specific research/engineering question before moving on.

---

## Phase 0 — Foundation & Proof of Concept

**Question**: *Can we backpropagate a topological loss through LoRA parameters at all?*

**Goal**: Validate that the core loop works end-to-end on a tiny model. No performance targets — just prove the gradient flow is correct and PH computation is tractable.

### Tasks

- [x] Project structure, packaging, CI basics
- [x] LoRA adapter wrapper with checkpoint/restore (PEFT-based)
- [x] Differentiable PH loss module (GUDHI backend)
  - [x] Persistence maximization loss
  - [x] Betti number targeting loss
  - [x] Diagram divergence loss (adapted vs. reference)
- [x] Core TTT loop implementation
- [x] Toy experiment script (GPT-2 on CPU)
- [x] Unit tests for adapter + PH loss
- [ ] **Validate**: run `toy_ttt.py`, confirm loss decreases, LoRA weights change
- [ ] **Benchmark**: measure PH computation time vs. activation size (sweep `max_points` from 16 to 512)
- [ ] Document findings: what works, what's slow, gradient quality observations

### Key risks to probe
- PH gradients may be sparse/noisy — does the loss actually decrease?
- GUDHI's numpy detour breaks the autograd graph for some operations — verify gradient flow
- Memory: how large can activation point clouds be before PH becomes intractable?

---

## Phase 1 — Real Model Validation

**Question**: *Does topological refinement produce measurably better outputs on a real LLM?*

**Goal**: Move from GPT-2 toy to a real instruction-following model (Gemma-2-2B or TinyLlama-1.1B). Define evaluation metrics and run controlled experiments.

### Tasks

- [ ] Port to Gemma-2-2B-it or TinyLlama-1.1B (GPU required)
- [ ] Design evaluation protocol:
  - Pick 3-5 tasks where "topological fidelity" could plausibly help (e.g., structured reasoning, code generation, factual consistency)
  - Define proxy metrics (perplexity, task accuracy, output coherence scores)
  - Compare: base model vs. static LoRA vs. LiveLoRA (TTT-refined)
- [ ] Profile latency: measure full TTT loop time per input on consumer GPU (RTX 3090/4090)
  - Target: <200ms total overhead for 3 TTT steps
- [ ] Experiment with PH loss variants:
  - Which homological dimensions matter (H0 only? H0+H1?)?
  - Which layers' activations to use (last layer? middle? multiple)?
  - What `max_points` gives the best quality/speed tradeoff?
- [ ] Add Weights & Biases logging to experiments

### Expected outcomes
- Clarity on whether topological loss improves outputs vs. random LoRA perturbation
- Latency profile to determine if real-time use is feasible
- Understanding of which PH features (dimensions, layers) carry signal

---

## Phase 2 — ScaleNet & Dynamic Modulation

**Question**: *Can learned per-layer LR modulation improve TTT convergence?*

**Goal**: Train a ScaleNet that predicts optimal per-layer learning rates from the topological signal, replacing fixed LR for all layers.

### Tasks

- [ ] Implement ScaleNet-modulated optimizer (per-layer LR = base_lr * scale_factor)
- [ ] Data collection: run Phase 1 experiments with per-layer gradient logging
- [ ] Train ScaleNet on collected data (meta-learning: inner loop = TTT, outer loop = ScaleNet)
- [ ] Compare convergence: fixed LR vs. ScaleNet-modulated, across inputs
- [ ] Explore surprise-proportional variants (Titans-style: scale ∝ gradient magnitude)
- [ ] Ablation: how many ScaleNet params are needed? (tiny MLP vs. per-layer scalar)

### Dependencies
- Requires solid Phase 1 results showing TTT actually helps

---

## Phase 3 — Warm-Start Hypernetwork

**Question**: *Can a hypernetwork predict a good initial LoRA, reducing TTT steps needed?*

**Goal**: Train a Perceiver-style hypernetwork that maps input activations to initial LoRA weights, providing a better starting point than zero/random initialization.

### Tasks

- [ ] Implement Perceiver cross-attention hypernetwork (inspired by Doc-to-LoRA architecture)
- [ ] Define training objective: minimize TTT loss after 0 steps (i.e., the hypernetwork output should already be a good adapter)
- [ ] Collect training data: (input, optimal-LoRA) pairs from Phase 1/2 experiments
- [ ] Train on small corpus, evaluate:
  - Does warm-start reduce required TTT steps (e.g., from 3 to 1)?
  - Does it improve final quality vs. zero-init + more steps?
- [ ] Profile: hypernetwork forward pass latency (target: <50ms)

### Dependencies
- Requires understanding from Phase 2 of what "good LoRA weights" look like per-input

---

## Phase 4 — Serving & Integration

**Question**: *Can this run in a production-like inference pipeline?*

**Goal**: Build adapter management infrastructure for dynamic per-instance LoRA serving.

### Tasks

- [ ] Adapter manager: pool of active LoRA adapters with LRU eviction
- [ ] Session-persistent LoRA: maintain adapter state across a conversation (not just per-query)
- [ ] Integration with vLLM or SGLang for batched serving
- [ ] Memory budget system: track LoRA memory usage, enforce limits
- [ ] API design: simple interface for "send prompt, get response with TTT"

### Dependencies
- Requires a validated, fast TTT loop from Phases 1-2

---

## Phase 5 — Advanced Topology & Research

**Question**: *What topological signals are most useful, and can we go beyond basic PH?*

These are research explorations once the core system is solid.

### Ideas to investigate

- [ ] **Cubical persistence** for image/vision model activations (faster than Rips for grid data)
- [ ] **Multi-scale PH**: compute PH at multiple filtration scales, weight loss by scale
- [ ] **Topological attention**: use PH features to modulate attention weights directly
- [ ] **Cross-layer topology**: compute PH on activations concatenated across layers (not per-layer)
- [ ] **Task-specific topological priors**: learn target Betti numbers per task type
- [ ] **Continuous LoRA rank adaptation**: dynamically adjust LoRA rank based on topological complexity
- [ ] **Online-LoRA with Laplace**: use Laplace approximation regularization for continuous streams (per Online-LoRA, WACV 2025)

---

## Non-Goals (for now)

- **Training from scratch**: LiveLoRA is an inference-time system, not a training method
- **Multi-GPU / distributed**: Focus on single-GPU first
- **Production deployment**: Phase 4 is about feasibility, not production hardening
- **Vision models**: Start with language models only; vision is a future extension

---

## How to Contribute

Pick a task from the current phase (Phase 0 right now). The most valuable contributions are:

1. **Running experiments** and reporting results (especially GPU benchmarks)
2. **Alternative PH backends** (TopologyLayer, Ripser++ bindings) for speed comparison
3. **Evaluation protocols** for measuring whether TTT actually helps on downstream tasks
