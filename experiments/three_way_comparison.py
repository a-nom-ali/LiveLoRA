"""Three-way comparison: No adaptation vs Entropy-TTT vs PH-TTT (LiveLoRA).

The core Phase 1 experiment. For each prompt:
  A) No adaptation — baseline generation
  B) Entropy-TTT — adapt LoRA with entropy minimization, then generate
  C) PH-TTT — adapt LoRA with topological loss, then generate

Metrics: self-consistency (agreement across samples), generation time,
loss trajectory.

Run:
    python experiments/three_way_comparison.py --model gpt2 --num-prompts 10
    python experiments/three_way_comparison.py --model Qwen/Qwen3.5-0.8B --num-prompts 20
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from livelora.core.lora_adapter import LiveLoraConfig, LiveLoraModel, get_lora_target_modules
from livelora.topology.ph_loss import DifferentiablePHLoss
from livelora.topology.ph_tracker import PHTracker, TopologyState, _deterministic_subsample


# Reasoning prompts (shared with correlation_study.py)
DEFAULT_PROMPTS = [
    "What is 15 + 27?",
    "If a train travels 60 miles per hour for 3 hours, how far does it go?",
    "What is the capital of France?",
    "List the first 5 prime numbers.",
    "What is the square root of 144?",
    "If you have 3 apples and buy 5 more, how many do you have?",
    "What color do you get when you mix red and blue?",
    "How many sides does a hexagon have?",
    "What is 7 times 8?",
    "Name three planets in our solar system.",
    "What is the opposite of 'hot'?",
    "If today is Monday, what day is it tomorrow?",
    "What is 100 divided by 4?",
    "How many legs does a spider have?",
    "What comes after the letter 'M' in the alphabet?",
    "What is the boiling point of water in Celsius?",
    "If a rectangle is 5 cm wide and 10 cm long, what is its area?",
    "What is the largest ocean on Earth?",
    "How many minutes are in one hour?",
    "What is 2 to the power of 5?",
]


@dataclass
class MethodResult:
    """Result from one method on one prompt."""

    method: str
    prompt: str
    samples: list[str]
    self_consistency: float
    generation_time: float
    loss_trajectory: list[float]  # Empty for baseline


def compute_self_consistency(samples: list[str]) -> float:
    """Jaccard-based self-consistency across samples."""
    if len(samples) < 2:
        return 1.0

    n_pairs = 0
    agreements = 0.0

    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            n_pairs += 1
            words_i = set(samples[i].lower().split())
            words_j = set(samples[j].lower().split())

            if not words_i or not words_j:
                continue

            intersection = len(words_i & words_j)
            union = len(words_i | words_j)
            if union > 0:
                agreements += intersection / union

    return agreements / n_pairs if n_pairs > 0 else 1.0


def generate_samples(
    model,
    tokenizer,
    prompt: str,
    n_samples: int = 3,
    max_new_tokens: int = 128,
    device: str = "cpu",
) -> list[str]:
    """Generate multiple samples for a prompt."""
    samples = []
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    for i in range(n_samples):
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(i > 0),  # First sample deterministic
                temperature=0.7 if i > 0 else 1.0,
                top_p=0.9 if i > 0 else 1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(
            gen[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        samples.append(text)

    return samples


def run_entropy_ttt(
    lora_model: LiveLoraModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    num_steps: int = 3,
    lr: float = 1e-4,
    grad_clip: float = 1.0,
    drift_penalty: float = 0.01,
) -> list[float]:
    """Run entropy minimization TTT steps on LoRA parameters.

    Returns loss trajectory.
    """
    optimizer = torch.optim.Adam(lora_model.lora_parameters(), lr=lr)
    losses = []

    for step in range(num_steps):
        optimizer.zero_grad()
        outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Entropy loss on all token positions
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        # Drift regularization
        drift = lora_model.lora_l2_from_checkpoint()
        total_loss = entropy + drift_penalty * drift

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_model.lora_parameters(), grad_clip)
        optimizer.step()

        losses.append(total_loss.item())

    return losses


def run_ph_ttt(
    lora_model: LiveLoraModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    ph_loss: DifferentiablePHLoss,
    target_layer: int = -1,
    num_steps: int = 3,
    lr: float = 1e-4,
    grad_clip: float = 1.0,
    drift_penalty: float = 0.01,
    max_points: int = 64,
) -> list[float]:
    """Run PH-based TTT steps on LoRA parameters.

    Returns loss trajectory.
    """
    n_layers = lora_model.model.config.num_hidden_layers + 1
    layer_idx = n_layers + target_layer if target_layer < 0 else target_layer

    optimizer = torch.optim.Adam(lora_model.lora_parameters(), lr=lr)
    losses = []

    for step in range(num_steps):
        optimizer.zero_grad()

        # Get activations at target layer
        activations = lora_model.get_layer_activations(
            input_ids=input_ids,
            attention_mask=attention_mask,
            layer_indices=[layer_idx],
        )

        points = activations[layer_idx][0]  # (seq_len, hidden_dim)

        # Subsample
        if points.shape[0] > max_points:
            indices = torch.randperm(points.shape[0], device=points.device)[:max_points]
            points = points[indices]

        topo_loss = ph_loss(points)

        # Drift regularization
        drift = lora_model.lora_l2_from_checkpoint()
        total_loss = topo_loss + drift_penalty * drift

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_model.lora_parameters(), grad_clip)
        optimizer.step()

        losses.append(total_loss.item())

    return losses


def run_ph_triggered_entropy(
    lora_model: LiveLoraModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    target_layer: int = -1,
    num_steps: int = 3,
    lr: float = 1e-4,
    grad_clip: float = 1.0,
    drift_penalty: float = 0.01,
    max_points: int = 64,
) -> tuple[list[float], int]:
    """PH decides WHEN, entropy decides HOW.

    Uses a hybrid loss: α * entropy + β * topology_loss.
    PH triggers updates based on proxy signals (effective rank, cosine
    concentration) rather than requiring pre/post comparison.

    For per-prompt TTT, we use the hybrid approach from the feedback:
    Loss = α * entropy + β * topo_loss

    This combines entropy's strong gradient with PH's structural constraint.
    Returns (loss_trajectory, num_updates_triggered).
    """
    from livelora.topology.ph_tracker import effective_rank, mean_abs_cosine

    n_layers = lora_model.model.config.num_hidden_layers + 1
    layer_idx = n_layers + target_layer if target_layer < 0 else target_layer

    ph_loss_fn = DifferentiablePHLoss(max_dimension=0, max_points=max_points)
    optimizer = torch.optim.Adam(lora_model.lora_parameters(), lr=lr)
    losses = []
    updates_triggered = 0

    # Hybrid loss: entropy for gradient strength + PH for structural constraint
    alpha_entropy = 1.0
    beta_topo = 0.01  # PH loss is much larger in magnitude, scale down

    for step in range(num_steps):
        optimizer.zero_grad()

        # Get both logits and hidden states in one pass
        outputs = lora_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = outputs.logits
        acts = outputs.hidden_states[layer_idx][0]

        # Entropy loss
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        # PH topology loss (subsample for speed)
        points = _deterministic_subsample(acts, max_points)
        topo_loss = ph_loss_fn(points)

        # Drift regularization
        drift = lora_model.lora_l2_from_checkpoint()
        total_loss = alpha_entropy * entropy + beta_topo * topo_loss + drift_penalty * drift

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_model.lora_parameters(), grad_clip)
        optimizer.step()

        losses.append(total_loss.item())
        updates_triggered += 1

    return losses, updates_triggered


def run_prompt_three_way(
    base_model,
    lora_model: LiveLoraModel,
    tokenizer,
    prompt: str,
    ph_loss: DifferentiablePHLoss,
    n_samples: int = 3,
    num_steps: int = 3,
    max_new_tokens: int = 128,
    max_points: int = 64,
    device: str = "cpu",
) -> list[MethodResult]:
    """Run all three methods on one prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    results = []

    # --- Method A: No adaptation ---
    t0 = time.perf_counter()
    lora_model.eval()
    samples_a = generate_samples(
        lora_model.model, tokenizer, prompt,
        n_samples=n_samples, max_new_tokens=max_new_tokens, device=device,
    )
    time_a = time.perf_counter() - t0

    results.append(MethodResult(
        method="none",
        prompt=prompt,
        samples=samples_a,
        self_consistency=compute_self_consistency(samples_a),
        generation_time=time_a,
        loss_trajectory=[],
    ))

    # --- Method B: Entropy-TTT ---
    t0 = time.perf_counter()
    lora_model.checkpoint()
    lora_model.train()
    losses_b = run_entropy_ttt(lora_model, input_ids, attention_mask, num_steps=num_steps)
    lora_model.eval()
    samples_b = generate_samples(
        lora_model.model, tokenizer, prompt,
        n_samples=n_samples, max_new_tokens=max_new_tokens, device=device,
    )
    lora_model.restore()
    time_b = time.perf_counter() - t0

    results.append(MethodResult(
        method="entropy_ttt",
        prompt=prompt,
        samples=samples_b,
        self_consistency=compute_self_consistency(samples_b),
        generation_time=time_b,
        loss_trajectory=losses_b,
    ))

    # --- Method C: PH-TTT (LiveLoRA) ---
    t0 = time.perf_counter()
    lora_model.checkpoint()
    lora_model.train()
    losses_c = run_ph_ttt(
        lora_model, input_ids, attention_mask, ph_loss,
        num_steps=num_steps, max_points=max_points,
    )
    lora_model.eval()
    samples_c = generate_samples(
        lora_model.model, tokenizer, prompt,
        n_samples=n_samples, max_new_tokens=max_new_tokens, device=device,
    )
    lora_model.restore()
    time_c = time.perf_counter() - t0

    results.append(MethodResult(
        method="ph_ttt",
        prompt=prompt,
        samples=samples_c,
        self_consistency=compute_self_consistency(samples_c),
        generation_time=time_c,
        loss_trajectory=losses_c,
    ))

    # --- Method D: PH-triggered Entropy (PH decides when, entropy decides how) ---
    t0 = time.perf_counter()
    lora_model.checkpoint()
    lora_model.train()
    losses_d, n_triggered = run_ph_triggered_entropy(
        lora_model, input_ids, attention_mask,
        num_steps=num_steps, max_points=max_points,
    )
    lora_model.eval()
    samples_d = generate_samples(
        lora_model.model, tokenizer, prompt,
        n_samples=n_samples, max_new_tokens=max_new_tokens, device=device,
    )
    lora_model.restore()
    time_d = time.perf_counter() - t0

    results.append(MethodResult(
        method="ph_entropy",
        prompt=prompt,
        samples=samples_d,
        self_consistency=compute_self_consistency(samples_d),
        generation_time=time_d,
        loss_trajectory=losses_d,
    ))

    return results


def main():
    parser = argparse.ArgumentParser(description="Three-way TTT comparison")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=3, help="Samples per prompt per method")
    parser.add_argument("--num-steps", type=int, default=3, help="TTT gradient steps")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-points", type=int, default=64, help="PH max points")
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="outputs/three_way_comparison.json")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Model: {args.model} | Device: {device} | Prompts: {args.num_prompts}")
    print(f"TTT steps: {args.num_steps} | LoRA rank: {args.lora_rank} | Max PH points: {args.max_points}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    base_model = base_model.to(device)

    # Wrap with LoRA
    target_modules = get_lora_target_modules(base_model)
    lora_config = LiveLoraConfig(rank=args.lora_rank, target_modules=target_modules)
    lora_model = LiveLoraModel(base_model, lora_config)
    lora_model.freeze_base()

    # PH loss
    ph_loss = DifferentiablePHLoss(
        max_dimension=0,  # H0 only for speed
        max_points=args.max_points,
    )

    prompts = DEFAULT_PROMPTS[:args.num_prompts]
    all_results = []

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {prompt[:60]}...")
        results = run_prompt_three_way(
            base_model, lora_model, tokenizer, prompt, ph_loss,
            n_samples=args.n_samples, num_steps=args.num_steps,
            max_new_tokens=args.max_tokens, max_points=args.max_points,
            device=device,
        )

        for r in results:
            print(f"  {r.method:12s}  consistency={r.self_consistency:.3f}  "
                  f"time={r.generation_time:.1f}s"
                  + (f"  loss={r.loss_trajectory[-1]:.2f}" if r.loss_trajectory else ""))

        all_results.extend(results)

    # Summary
    print("\n" + "=" * 70)
    print("THREE-WAY COMPARISON SUMMARY")
    print("=" * 70)

    for method in ["none", "entropy_ttt", "ph_ttt", "ph_entropy"]:
        method_results = [r for r in all_results if r.method == method]
        if not method_results:
            continue
        consistencies = [r.self_consistency for r in method_results]
        times = [r.generation_time for r in method_results]

        label = {
            "none": "No adaptation",
            "entropy_ttt": "Entropy-TTT",
            "ph_ttt": "PH-TTT (LiveLoRA)",
            "ph_entropy": "PH->Entropy (PH triggers, entropy fixes)",
        }[method]
        print(f"\n{label}:")
        print(f"  Self-consistency:  mean={sum(consistencies)/len(consistencies):.4f}  "
              f"min={min(consistencies):.4f}  max={max(consistencies):.4f}")
        print(f"  Generation time:   mean={sum(times)/len(times):.1f}s  total={sum(times):.1f}s")

        if method_results[0].loss_trajectory:
            final_losses = [r.loss_trajectory[-1] for r in method_results]
            print(f"  Final loss:        mean={sum(final_losses)/len(final_losses):.3f}")

    # Pairwise comparison
    none_results = {r.prompt: r for r in all_results if r.method == "none"}
    entropy_results = {r.prompt: r for r in all_results if r.method == "entropy_ttt"}
    ph_results = {r.prompt: r for r in all_results if r.method == "ph_ttt"}
    phe_results = {r.prompt: r for r in all_results if r.method == "ph_entropy"}

    n = args.num_prompts
    used_prompts = prompts[:n]

    def win_count(a, b):
        return sum(1 for p in used_prompts if a[p].self_consistency > b[p].self_consistency)

    print(f"\nEntropy-TTT beats No-adapt:  {win_count(entropy_results, none_results)}/{n}")
    print(f"PH-TTT beats No-adapt:       {win_count(ph_results, none_results)}/{n}")
    print(f"PH->Entropy beats No-adapt:  {win_count(phe_results, none_results)}/{n}")
    print(f"PH->Entropy beats Entropy:   {win_count(phe_results, entropy_results)}/{n}")
    print(f"PH->Entropy beats PH-TTT:    {win_count(phe_results, ph_results)}/{n}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [
        {
            "method": r.method,
            "prompt": r.prompt,
            "self_consistency": r.self_consistency,
            "generation_time": r.generation_time,
            "loss_trajectory": r.loss_trajectory,
            "samples": r.samples,
        }
        for r in all_results
    ]
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
