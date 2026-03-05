## LiveLoRA Update Feedback: Topology-Faithful Stabilization via MDL Ratio Gate + KL Trust Region

### Goal

Make inference-time LoRA updates **topology-faithful** (structural repair) rather than **conversation-faithful** (micro fine-tuning on the current exchange). Achieve stability by separating:

* **Structural improvement** (topology + minimal parameter drift)
* **Semantic drift** (changes to output distribution)

…and only accepting updates with high structural gain per unit semantic movement.

---

## 1) Core Formulation

### Observation surface (per chunk)

For a token chunk (t_{i..i+N}), extract activations (H_\theta \in \mathbb{R}^{N \times d}) from 1–few selected layers (recommend mid-layers first).

Compute a topological summary (\pi(H_\theta)) (persistence landscape / Betti curve / diagram summary).

### Topology target (\pi^\star) (pick one)

**Phase 0 recommendation (self-consistency):**
[
\pi^\star := \pi(H_{\theta_0})
]
i.e., compute topology at the warm-start/checkpoint LoRA state (\theta_0) and treat this as the “healthy” baseline for the current query/chunk. This makes updates *repair drift* rather than “learn content.”

**Later (population prior):**
[
\pi^\star := \text{prior}(\cdot)
]
learned offline from good reasoning traces.

---

## 2) Losses

### Structural loss (what we optimize)

[
L_{\text{struct}}(\theta) = D_{\text{topo}}(\theta) + \lambda |\theta-\theta_0|*2^2
]
where:
[
D*{\text{topo}}(\theta)=d(\pi(H_\theta), \pi^\star)
]
Use a differentiable (d): landscape (L_2) tends to be stable; Wasserstein on diagrams is powerful but can be noisier.

### Semantic drift (what we constrain)

Anchor semantics to the pre-update distribution:
[
L_{\text{sem}}(\theta)=\mathrm{KL}\big(p_{\theta_0}(\cdot|x);||;p_{\theta}(\cdot|x)\big)
]
This explicitly prevents the adapter from “learning the conversation.”

---

## 3) Compression-Ratio Signal + Accept/Reject Gate

Compute before/after candidate update:
[
\Delta L_{\text{struct}} = L_{\text{struct}}(\theta_{\text{before}})-L_{\text{struct}}(\theta_{\text{after}})
]
[
\Delta L_{\text{sem}} = L_{\text{sem}}(\theta_{\text{after}})-L_{\text{sem}}(\theta_{\text{before}})
]

Define the stability ratio:
[
\rho = \frac{\Delta L_{\text{struct}}}{\Delta L_{\text{sem}}+\beta}
]
((\beta) small constant to avoid division issues)

**Accept update iff:**

1. **Trust region:** (L_{\text{sem}}(\theta_{\text{after}}) \le \varepsilon)
2. **Payoff gate:** (\rho \ge \tau)
3. **Must improve:** (\Delta L_{\text{struct}} > 0)

Interpretation: only accept updates that buy a lot of structural/topological improvement for negligible semantic displacement.

---

## 4) Drop-in Pseudocode (Chunk Update Step)

```python
def livelora_chunk_update(
    model, lora, x_chunk,
    theta0_snapshot,
    topo_target,                  # pi* (self-consistency) or learned prior target
    lambda_rate=1e-2,
    eps_kl=1e-4,
    tau_rho=50.0,
    beta=1e-8,
    lr=1e-4,
    grad_clip=1.0
):
    """
    One-step candidate update on LoRA, then accept/reject using:
      - KL trust region (semantic pinning)
      - rho ratio gate (structural gain per semantic drift)
    """

    # ---- (0) Baseline forward (no update) ----
    with torch.no_grad():
        out0 = model.forward_with_activations(x_chunk, lora_state="current")
        H0 = out0.activations          # N x d (chosen layer(s))
        p0 = out0.token_dists.detach() # dist anchor for KL

    # Baseline topology + losses
    pi0 = topo_summary(H0)
    D_topo0 = topo_distance(pi0, topo_target)
    R0 = lora.l2_distance_to(theta0_snapshot)    # ||theta - theta0||^2
    L_struct0 = D_topo0 + lambda_rate * R0
    L_sem0 = 0.0

    # ---- (1) Candidate update: one gradient step on LoRA only ----
    lora.save_candidate_state()

    out = model.forward_with_activations(x_chunk, lora_state="current")
    H = out.activations
    p = out.token_dists

    pi = topo_summary(H)
    D_topo = topo_distance(pi, topo_target)
    R = lora.l2_distance_to(theta0_snapshot)
    L_struct = D_topo + lambda_rate * R

    lora.zero_grad()
    L_struct.backward()
    torch.nn.utils.clip_grad_norm_(lora.parameters(), grad_clip)
    lora.step(lr=lr)

    # ---- (2) Evaluate candidate ----
    with torch.no_grad():
        out1 = model.forward_with_activations(x_chunk, lora_state="current")
        H1 = out1.activations
        p1 = out1.token_dists.detach()

        pi1 = topo_summary(H1)
        D_topo1 = topo_distance(pi1, topo_target)
        R1 = lora.l2_distance_to(theta0_snapshot)
        L_struct1 = D_topo1 + lambda_rate * R1

        KL = kl_divergence(p0, p1)  # avg over tokens in chunk

        d_struct = float((L_struct0 - L_struct1).item())
        d_sem = float(KL - L_sem0)
        rho = d_struct / (d_sem + beta)

    # ---- (3) Accept / reject ----
    accept = (KL <= eps_kl) and (rho >= tau_rho) and (d_struct > 0.0)

    if not accept:
        lora.restore_candidate_state()

    metrics = {
        "D_topo0": float(D_topo0.item()),
        "D_topo1": float(D_topo1.item()),
        "R0": float(R0),
        "R1": float(R1),
        "L_struct0": float(L_struct0.item()),
        "L_struct1": float(L_struct1.item()),
        "KL": float(KL),
        "d_struct": float(d_struct),
        "rho": float(rho),
        "accepted": bool(accept),
    }

    return accept, metrics
```

---

## 5) Implementation Notes / Defaults That Usually Behave

### Where to adapt

* Apply LoRA to **Q/K/V** (and optionally O) in **2–4 mid layers**, not all layers.
* Rationale: Q/K shape the token graph (topology operator), V controls information flow.

### Chunking

* Run this per chunk (16–64 tokens) rather than per token.

### Trust-region + ratio tuning

* Start with (\varepsilon \in [10^{-4}, 10^{-3}]) avg KL per chunk.
* Choose (\tau) so only “rare but high payoff” updates pass (expect acceptance rate to be low at first).

### Why this avoids “conversation learning”

* (L_2) drift regularizer alone limits parameter movement but doesn’t prevent *semantic drift*.
* KL anchoring explicitly pins outputs close to (\theta_0), so updates are forced to act as geometry repair.

---

## 6) How This Differs from Current LiveLoRA (delta)

LiveLoRA already includes:

* TTT loop + PH loss
* adapt-and-reset
* L2 drift regularization + gradient clipping
* (planned) ScaleNet modulation

**New additions:**

1. **Explicit semantic pinning** via KL trust region
   → prevents “learning the conversation.”

2. **Accept/reject ratio gate ((\rho))**
   → update is only kept if structural improvement is high relative to semantic movement.

3. **Clear decomposition of change**
   → structure vs meaning becomes measurable, not implied.

This converts LiveLoRA into a **controlled structural repair mechanism** rather than a general per-instance fine-tuning loop.

---

## 7) Mapping to General Relativity of Change (GRC)

This yields a clean GRC-style decomposition:

* **Structural change field:** (\Delta L_{\text{struct}})
  (topology distortion reduction + cost of parameter movement)

* **Semantic displacement:** (\Delta L_{\text{sem}})
  (KL drift in output distribution)

* **Change efficiency / curvature control:**
  [
  \rho = \frac{\Delta L_{\text{struct}}}{\Delta L_{\text{sem}}+\beta}
  ]
  Interpretation: “allow change only when it increases structural coherence without warping the expressed trajectory.”

---

## 8) Suggested Next Enhancements (optional)

1. **Hybrid proxy → PH escalation:**
   Use cheap attention/graph proxies most of the time; compute PH only when proxies trigger.

2. **Fisher/Laplace drift penalty:**
   Replace (|\theta-\theta_0|^2) with a diagonal Fisher/Laplace penalty for stronger stability under repeated updates.

---

**Summary:** Add a KL trust region + (\rho) accept/reject gate to force LiveLoRA updates to behave as topology repair (structural compression) rather than conversation adaptation, aligning cleanly with the GRC “change field vs displacement” intuition.
