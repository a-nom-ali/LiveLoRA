# LiveLoRA Roadmap

Organized into phases, each answering a specific research question before moving on.

---

## Phase 0 — Foundation & Proof of Concept ✅ COMPLETE

**Question**: *Can we backpropagate a topological loss through LoRA parameters at all?*

### Completed
- [x] Project structure, packaging, tests (40 tests, ~6s)
- [x] LoRA adapter wrapper with checkpoint/restore (PEFT-based)
- [x] Differentiable PH loss module (GUDHI backend) — persistence, Betti, divergence modes
- [x] Core TTT loop implementation
- [x] Toy experiment script (Qwen3.5-0.8B, GPT-2)
- [x] ChatGPT export data loader
- [x] Auto target module detection across architectures
- [x] **Validated**: loss decreases, LoRA weights change, gradients flow in 100% of cases
- [x] **Benchmarked**: 64 points ~25-65ms (sweet spot), H0-only 10x faster than H0+H1

### Key findings
- PCA before PH does NOT help speed (Rips complex dominates)
- 64 points optimal, 128 borderline, 256+ intractable
- bf16 on CUDA requires float32 cast for cdist, eigvalsh, numpy

---

## Phase 1 — Real Model Validation ✅ STRONG RESULTS

**Question**: *Does topological refinement produce measurably better outputs?*

### Completed
- [x] **Correlation study** (Qwen3.5-0.8B, n=20): Spearman ρ = -0.39 — topology predicts quality
- [x] **4-way per-prompt comparison** (Qwen3.5-0.8B, GPU):
  - Hybrid (entropy+PH): mean=0.307, wins 14/20 vs entropy-only
  - PH-TTT: mean=0.268, wins 13/20 vs baseline
  - Entropy-TTT: mean=0.242
  - Baseline: mean=0.230
- [x] Entropy loss baseline (EntropyLoss, MarginalEntropyLoss)
- [x] Cheap proxy detectors: effective_rank(), mean_abs_cosine()
- [x] Deterministic subsampling for stable topology comparisons
- [x] Multi-position KL trust region (last 8 positions, not just final token)

### Key finding
- PH alone has weaker gradients than entropy, but adding PH as a structural constraint alongside entropy (hybrid loss) wins decisively in per-prompt TTT

### Not yet done
- [ ] Profile latency on consumer GPU
- [ ] Persistence landscapes for smoother gradients
- [ ] Layer selection experiments (which layers carry most signal)
- [ ] Weights & Biases logging
- [ ] Visualization (persistence diagrams, Betti curves, activation manifold projections)

---

## Phase 2 — LiveLoRA-Delta: Chunked Generation ✅ VALIDATED

**Question**: *Can topology-gated adaptation during generation improve output quality?*

### Completed
- [x] `gen_controller.py` — chunked generation with MDL ratio gate
  - Configurable chunk size, KL trust region, rho gate
  - Conditional PH escalation (STABLE=skip, DRIFTING=1 attempt, COLLAPSING=2)
  - State-dependent gate thresholds (COLLAPSING more permissive)
  - Three optimization modes: `ph`, `entropy`, `hybrid` + ablation modes
- [x] `ph_tracker.py` — rolling topology tracker
  - TopologySummary with Betti numbers, persistence stats, cheap proxies
  - Directional degradation scoring (only penalizes topology decrease)
  - Absolute divergence drift threshold (1.5x) for expansion detection
  - EMA baseline mode for slow-following baseline
- [x] **4-way delta comparison** (Qwen3.5-0.8B, n=20, GPU):
  - **PH-Delta: mean=0.874** — wins 20/20 vs baseline, 9-7 vs entropy, 8-5 vs hybrid
  - Entropy-Delta: mean=0.815, 100% acceptance
  - Hybrid-Delta: mean=0.796
  - Baseline: mean=0.255
- [x] Per-state diagnostics: acceptance rate by state, topo improvement rate

### Key finding
**Topology-constrained test-time adaptation improves stability by selectively admitting structurally beneficial updates.** PH's 39% acceptance rate is a feature — it acts as a structural quality filter that prevents entropy overfitting. All methods improve topology at similar rates when they update; PH wins by being selective.

### Gate ablation results (n=20, Qwen3.5-0.8B)

| Method | Consistency | Acceptance | Updates |
|---|---|---|---|
| **Entropy-grad + PH-gate** | **0.897** | 23% | 13/57 |
| Random + PH-gate | 0.889 | 17% | 10/60 |
| PH-grad + PH-gate | 0.827 | 60% | 29/48 |
| Entropy-budgeted (max=1) | 0.789 | 100% | 18/18 |
| Entropy-grad + topo-gate | 0.773 | 100% | 36/36 |
| Baseline | 0.235 | — | — |

**The PH gate is the hero, not the PH gradient.** Entropy+PH-gate and Random+PH-gate both beat PH-grad+PH-gate. The mechanism is selective rejection — lower acceptance rate correlates with higher consistency. Even random noise filtered by the structural gate produces strong results.

### Threshold sweep results (n=10, Qwen3.5-0.8B, entropy_ph_gate mode)

| tau_rho | Consistency | Acceptance |
|---------|------------|------------|
| 0.0     | 0.846      | 55%        |
| 1.0     | 0.803      | 45%        |
| 5.0     | 0.910      | 30%        |
| 10.0    | 0.887      | 43%        |
| 25.0    | 0.913      | 43%        |
| **50.0**| **0.941**  | **21%**    |
| 100.0   | 0.872      | 45%        |

Performance peaks at ~20% acceptance and drops at both extremes. This matches **MCMC acceptance rate theory** — the system behaves as a stochastic search in parameter space constrained by topology. The optimal regime (tau_rho 5-50) naturally produces 20-43% acceptance.

### GSM8K ground truth (n=20, Qwen3.5-0.8B)

| Method | Accuracy | Consistency |
|--------|----------|-------------|
| Baseline | 5% (1/20) | 0.278 |
| Entropy + PH-gate | 5% (1/20) | 0.955 |
| Entropy + topo-gate | 5% (1/20) | 1.000 |

**Identical accuracy, dramatically different consistency.** The 0.8B model is below the capability threshold for GSM8K — adaptation stabilizes outputs but can't improve correctness on tasks the model fundamentally can't solve. PH gating makes the model *consistently wrong* rather than *randomly wrong*.

### In progress
- [ ] **GSM8K on Qwen3.5-7B** — test correctness at a scale where the model can sometimes solve the task
- [ ] Integrate ScaleNet into gen_controller
- [ ] Track first_collapse_chunk vs error_chunk timing

---

## Phase 3 — Warm-Start Hypernetwork

**Question**: *Can a hypernetwork predict a good initial LoRA, reducing TTT steps needed?*

### Tasks
- [ ] Perceiver cross-attention hypernetwork (Doc-to-LoRA inspired)
- [ ] Training objective: minimize TTT loss after 0 steps
- [ ] Collect training data from Phase 2 experiments
- [ ] Profile: hypernetwork forward pass latency (target: <50ms)

---

## Phase 4 — Serving & Integration

**Question**: *Can this run in a production-like inference pipeline?*

### Tasks
- [ ] Adapter manager: pool of active LoRA adapters with LRU eviction
- [ ] Session-persistent LoRA across conversations
- [ ] Integration with vLLM or SGLang
- [ ] Memory budget system
- [ ] API design

---

## Phase 5 — Research Extensions

- **Branching LoRA States**: maintain 2-4 competing adapter states during generation, prune by topology — beam search with different adapter states
- **Task-Specific Topology Priors**: classification = separate components, reasoning = smooth trajectory, RAG = loops linking concepts
- **Hybrid Proxy Escalation**: cheap proxies for most chunks, full PH only on anomaly
- **Fisher/Laplace Drift Penalty**: replace L2 drift with Fisher-weighted penalty
- **GRC Framework**: generalize the structural/semantic decomposition to any adaptive inference system
- **Advanced PH**: persistence landscapes, cubical persistence, cross-layer topology

---

## Paper Framing

The emerging thesis:

> **LiveLoRA: Structural Metropolis Test-Time Adaptation**
>
> Topology-guided structural gating improves inference-time adaptation by selectively admitting only structurally beneficial updates, independent of the optimization signal. The system behaves like a Monte Carlo search in parameter space constrained by persistent homology — a stochastic search with structural filtering where the acceptance criterion matters more than the proposal distribution.

Key claims:
1. Persistent homology on activation point clouds provides a **structural admission signal** that decides which LoRA updates to accept
2. The PH gate is the mechanism — gate ablation shows Entropy+PH-gate (0.897) and Random+PH-gate (0.889) both outperform PH-grad+PH-gate (0.827)
3. Selective rejection is the key: lower acceptance rate → higher consistency. The gate's value is in what it *rejects*, not in the gradient that proposes updates
4. The same topological signal serves different roles: structural constraint in per-prompt TTT, quality filter in chunked generation

---

## Non-Goals (for now)

- Training from scratch (LiveLoRA is inference-time only)
- Multi-GPU / distributed
- Production deployment
- Vision models (language first)
