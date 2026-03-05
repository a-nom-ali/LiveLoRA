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
- [x] Toy experiment script (default: Qwen3.5-0.8B)
- [x] Unit tests for adapter + PH loss
- [x] ChatGPT export data loader for test conversations
- [x] Auto target module detection for LoRA across architectures
- [x] **Validate**: run `toy_ttt.py`, confirm loss decreases, LoRA weights change — **CONFIRMED** (GPT-2, loss -298 → -303 over 3 steps, weight norm 13.9 → 17.2)
- [x] **Benchmark**: measure PH computation time vs. activation size — **DONE**
  - 64 points: ~25-65ms (sweet spot)
  - 128 points: ~150-310ms (borderline)
  - 256+ points: 3+ seconds (intractable)
  - H0-only is 10x faster than H0+H1 (32ms vs 329ms at 128 points)
  - Hidden dim barely affects PH time — Rips complex dominates
  - Gradients flow in 100% of test cases
- [x] **PCA projection**: tested — does NOT help PH speed (distance matrix is fast, simplex tree is the bottleneck). PCA may still help gradient quality (untested)
- [ ] **Correlation study**: measure topology of activations on varied inputs, correlate with output quality — does topology predict quality *before* any adaptation?
- [x] Document findings (see above + CLAUDE.md)

### Key risks resolved
- ~~PH gradients may be sparse/noisy~~ — **Loss decreases reliably.** Persistence landscapes still worth exploring for smoother gradients.
- ~~GUDHI's numpy detour breaks autograd~~ — **Gradients flow in 100% of cases** through distance matrix indexing.
- ~~Memory: how large can activation point clouds be~~ — **64 points is practical, 128 borderline, 256+ no.**

---

## Phase 1 — Real Model Validation

**Question**: *Does topological refinement produce measurably better outputs on a real LLM?*

**Goal**: Run controlled experiments on Qwen3.5-0.8B/2B. The killer comparison (from feedback): **no adaptation vs. entropy-loss TTT vs. PH-loss TTT (LiveLoRA)**. If PH wins even slightly, it's a publishable result.

### Tasks

- [ ] **Three-way comparison** on reasoning benchmarks (GSM8K, ARC, BIG-Bench):
  1. No adaptation (baseline)
  2. Entropy-loss TTT (existing approach from LoRA-TTT)
  3. PH-loss TTT (LiveLoRA)
- [ ] Profile latency: measure full TTT loop time per input on consumer GPU
  - Target: <200ms total overhead for 3 TTT steps
- [ ] Experiment with PH loss variants:
  - Which homological dimensions matter (H0 only? H0+H1?)?
  - Which layers' activations to use (middle layers are likely more stable than early/late)
  - What `max_points` gives the best quality/speed tradeoff?
  - **Persistence landscapes** vs. raw diagrams for gradient smoothness
  - **Wasserstein distance** between diagrams as loss
- [ ] Test with ChatGPT conversation data as real-world input distribution
- [ ] Add Weights & Biases logging to experiments
- [ ] **Visualization**: plot persistence diagrams, Betti curves, activation manifold projections (PCA/UMAP) — these figures are essential for understanding and communicating results

### Expected outcomes
- Clarity on whether topological loss improves outputs vs. entropy-based TTT
- Latency profile to determine if real-time use is feasible
- Understanding of which PH features (dimensions, layers) carry signal

### What "healthy reasoning topology" looks like (hypothesis)
- One main connected component (coherence)
- A few persistent loops (cross-linking constraints)
- Not too many short-lived noisy features (confusion/drift)

---

## Phase 2 — LiveLoRA-Delta: Chunked Generation with MDL Ratio Gate

**Question**: *Can we adapt LoRA during generation using topology-faithful updates that repair structure without learning the conversation?*

**Goal**: Instead of adapting once per prompt, adapt at selected points during generation when topology destabilizes. Critically, use a **structural vs. semantic decomposition** to ensure updates are topology repair, not micro fine-tuning on the current exchange.

### Core formulation: the MDL ratio gate

Each candidate LoRA update is decomposed into:

- **Structural loss** `L_struct = D_topo(theta) + lambda * ||theta - theta_0||^2` — topological distance to target + parameter drift
- **Semantic loss** `L_sem = KL(p_theta_0 || p_theta)` — KL divergence of output distribution from pre-update state

The **stability ratio** measures efficiency of the update:
```
rho = delta_L_struct / (delta_L_sem + beta)
```

**Accept update only if ALL three hold:**
1. **KL trust region**: `L_sem <= epsilon` (semantic pinning — prevents conversation learning)
2. **Payoff gate**: `rho >= tau` (high structural gain per unit semantic drift)
3. **Must improve**: `delta_L_struct > 0` (topology actually got better)

### Topology target (pi-star)

- **Phase 0/1 (self-consistency)**: `pi* = pi(H_theta_0)` — topology at checkpoint state is the "healthy" baseline. Updates repair drift back toward this.
- **Later (population prior)**: `pi*` learned offline from good reasoning traces.

### Tasks

- [ ] Implement `core/gen_controller.py` — chunked generation controller:
  - Configurable chunk size (default: 32 tokens)
  - Candidate update step: compute gradient, apply, evaluate, accept/reject
  - KL trust region computation (avg KL per chunk tokens)
  - Rho ratio gate with configurable `epsilon`, `tau`, `beta`
  - Update cooldown: skip consecutive chunks unless loss is severe
- [ ] Implement `topology/ph_tracker.py` — rolling topology baseline:
  - Store persistence summary stats from earlier chunks
  - Compare current chunk's diagram to baseline (Wasserstein distance)
  - Detect: stable, destabilizing, or collapsing topology
- [ ] **Three-way comparison** on reasoning tasks:
  1. No adaptation
  2. Adapt once per prompt (Phase 1 approach)
  3. Adapt per chunk with MDL ratio gate (LiveLoRA-Delta)
- [ ] Measure: accuracy, acceptance rate, average PH update count per generation, latency overhead
- [ ] Integrate ScaleNet-modulated optimizer (per-layer LR = base_lr * scale_factor)
- [ ] Explore surprise-proportional variants (Titans-style: scale proportional to gradient magnitude)

### Stability constraints (all work together)
1. Tiny LR + single gradient step per update event
2. Elastic drift anchor (L2 toward initial adapter)
3. Small LoRA rank (4-8)
4. Update cooldown (skip consecutive chunks)
5. **KL trust region** — pin output distribution close to pre-update state
6. **Rho payoff gate** — reject updates with low structural return on semantic investment

### Suggested defaults
- `epsilon` (KL trust region): 1e-4 to 1e-3 avg KL per chunk
- `tau` (payoff threshold): ~50 (expect low acceptance rate initially — that's correct)
- `beta` (division safety): 1e-8
- LR: 1e-4, grad_clip: 1.0
- LoRA: Q/K/V (+ optionally O) on 2-4 mid layers

### Dependencies
- Requires solid Phase 1 results showing per-prompt TTT actually helps

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

## Phase 5 — Aspirational Research

**Question**: *What becomes possible when topology-guided inference-time adaptation actually works?*

These are ambitious research directions once the core system is validated.

### Branching LoRA States (Topology-Guided Beam Search)

The most exciting idea: for ambiguous problems, maintain **2-4 competing LoRA adapter states** during generation. Each branch adapts slightly differently, each gets a topology score + token logprob score, and periodically prune to the most topologically coherent branch. This is beam search where **the beams have different adapter states** — exploring multiple representation manifold paths simultaneously.

### Task-Specific Topology Priors

Move PH from unsupervised to **semantically constrained**:
- **Classification**: each class = separate connected component; maximize persistence of class clusters
- **Reasoning**: trajectory manifold should be smooth; penalize topological discontinuities
- **RAG**: retrieved embeddings should form loops linking concepts; reward H1 features connecting query to evidence

This turns topology into a semantic constraint, not just a structural one.

### Topological Memory Systems

LLM + LoRA substrate + topology feedback = a model that **reshapes its internal geometry per query**. This is closer to adaptive representation geometry — how biological neural systems behave. The model maintains a topological "self-model" of its representations and actively optimizes it.

### Hybrid Proxy Escalation

Use cheap attention-graph or gradient-norm proxies for most chunks; only compute full PH when proxies trigger an anomaly. This could cut PH computation by 80-90% while preserving the topological signal where it matters.

### Fisher/Laplace Drift Penalty

Replace the simple `||theta - theta_0||^2` drift regularizer with a diagonal Fisher or Laplace approximation penalty — stronger stability under repeated updates, better calibrated to which parameters matter most.

### General Relativity of Change (GRC) Framework

The MDL ratio gate naturally decomposes into a GRC-style formulation:
- **Structural change field**: `delta_L_struct` (topology distortion reduction + parameter cost)
- **Semantic displacement**: `delta_L_sem` (KL drift in output distribution)
- **Change efficiency**: `rho` = structural gain / semantic cost — "allow change only when it increases structural coherence without warping the expressed trajectory"

This may generalize beyond LiveLoRA to a principled framework for any inference-time adaptation system.

### Advanced PH Techniques

- [ ] **Persistence landscapes** for smoother, more stable gradients than raw diagrams
- [ ] **Cubical persistence** for image/vision model activations (faster than Rips for grid data)
- [ ] **Multi-scale PH**: compute PH at multiple filtration scales, weight loss by scale
- [ ] **Topological attention**: use PH features to modulate attention weights directly
- [ ] **Cross-layer topology**: compute PH on activations concatenated across layers (not per-layer)
- [ ] **Continuous LoRA rank adaptation**: dynamically adjust LoRA rank based on topological complexity
- [ ] **Online-LoRA with Laplace**: use Laplace approximation regularization for continuous streams (per Online-LoRA, WACV 2025)

---

## Paper Framing

If this works, the paper positioning is:

> **LiveLoRA: Topology-Faithful Test-Time Adaptation for Large Language Models**
>
> LoRA = parameter substrate. TTT = adaptation protocol. PH = structural signal. MDL ratio gate = change control.
>
> We introduce a new class of inference-time losses based on topological invariants of representation manifolds, with a principled accept/reject mechanism that decomposes adapter updates into structural improvement vs. semantic displacement.

The key claims:
1. Most representation learning optimizes distance/similarity/entropy/contrast. LiveLoRA optimizes **topological invariants** — a qualitatively different signal
2. The MDL ratio gate ensures updates are **topology repair**, not conversation adaptation — a controlled structural repair mechanism

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
