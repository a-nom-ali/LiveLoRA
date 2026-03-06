"""LiveLoRA-Delta comparison: chunked generation with topology-gated adaptation.

This is the definitive experiment for LiveLoRA-Delta — tests the three
optimization modes in the chunked generation setting where PH tracking
actually triggers DRIFTING/COLLAPSING during generation:

  1. Baseline — no adaptation
  2. PH-Delta — optimize PH loss, gate on structural improvement
  3. Entropy-Delta — optimize entropy loss, gate on topology improvement (PH->Entropy)
  4. Hybrid-Delta — optimize alpha*entropy + beta*topo, gate on topology improvement

Unlike the per-prompt three_way_comparison, here the PHTracker observes
each generation chunk and only triggers updates when topology destabilizes.

Run:
    python experiments/delta_comparison.py --model gpt2
    python experiments/delta_comparison.py --model Qwen/Qwen3.5-0.8B --device auto
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from livelora.core.gen_controller import DeltaConfig, GenerationController
from livelora.core.lora_adapter import LiveLoraConfig, LiveLoraModel, get_lora_target_modules


PROMPTS = [
    "Explain step by step how to solve: What is 15 + 27 * 3?",
    "Write a short story about a robot learning to paint.",
    "If a train travels 60 miles per hour for 3.5 hours, then slows to 40 mph for 2 hours, what is the total distance?",
    "List the first 10 prime numbers and explain why each is prime.",
    "What are the three laws of thermodynamics? Explain each briefly.",
    "Describe the process of photosynthesis in simple terms.",
    "Write a Python function to check if a number is a palindrome, then trace through it with 12321.",
    "Compare and contrast cats and dogs as pets, giving three advantages of each.",
    "If you have 5 red balls and 3 blue balls in a bag, what is the probability of drawing 2 red balls in a row?",
    "Explain why the sky is blue using physics.",
    "What causes seasons on Earth? Explain the role of axial tilt.",
    "Write a haiku about autumn, then explain the syllable structure.",
    "If a rectangle has a perimeter of 30 cm and a width of 5 cm, what is its area?",
    "Describe the water cycle in four steps.",
    "What is the difference between DNA and RNA? List three differences.",
    "Explain how a binary search algorithm works with an example.",
    "If you invest $1000 at 5% annual interest compounded yearly, how much do you have after 3 years?",
    "Write a brief argument for and against renewable energy.",
    "What are Newton's three laws of motion? Give a real-world example for each.",
    "Explain the concept of supply and demand with a simple example.",
]


def create_lora_model(base_model):
    """Create a LiveLoraModel with LoRA adapters (wraps base_model in-place)."""
    target_modules = get_lora_target_modules(base_model)
    lora_config = LiveLoraConfig(
        rank=8,
        alpha=16.0,
        target_modules=target_modules,
        dropout=0.0,
        bias="none",
    )
    lora_model = LiveLoraModel(base_model, lora_config)
    lora_model.freeze_base()
    return lora_model


def run_baseline(model, tokenizer, prompt, max_new_tokens, device):
    """Generate without any adaptation."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def run_delta_method(
    lora_model: LiveLoraModel,
    tokenizer,
    prompt: str,
    optimization_mode: str,
    max_new_tokens: int,
    chunk_size: int,
    device: str,
) -> dict:
    """Run one prompt through LiveLoRA-Delta with given optimization mode.

    The lora_model is reused across calls — GenerationController.generate()
    handles checkpoint/restore internally.
    """
    # Gate parameters tuned per mode:
    #   PH: structural rho is large (100s), need tight KL to prevent drift
    #   Entropy/Hybrid: topology-divergence rho is small, need looser KL
    if optimization_mode == "ph":
        tau_rho = 10.0
        epsilon_kl = 0.01  # PH causes larger KL than entropy
    else:
        tau_rho = 0.0  # Accept any topology improvement
        epsilon_kl = 0.01

    config = DeltaConfig(
        chunk_size=chunk_size,
        max_new_tokens=max_new_tokens,
        target_layers=[-1],
        max_points=64,
        max_dimension=0,  # H0 only for speed
        lr=1e-4,
        epsilon_kl=epsilon_kl,
        tau_rho=tau_rho,
        cooldown_chunks=1,
        max_updates=5,
        conditional_ph=True,
        optimization_mode=optimization_mode,
        alpha_entropy=1.0,
        beta_topo=0.01,
        entropy_probe_len=8,
    )

    controller = GenerationController(lora_model, config)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    t0 = time.perf_counter()

    output_ids, metrics = controller.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        pad_token_id=tokenizer.pad_token_id,
    )

    elapsed = time.perf_counter() - t0

    generated_text = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    # Aggregate metrics
    n_accepted = sum(1 for m in metrics if m.accepted)
    n_attempted = sum(1 for m in metrics if m.attempts > 0)
    n_stable = sum(1 for m in metrics if m.reason == "stable_skip")
    n_drifting = sum(1 for m in metrics if m.topology_state == "drifting")
    n_collapsing = sum(1 for m in metrics if m.topology_state == "collapsing")
    mean_kl = sum(m.kl_divergence for m in metrics if m.attempts > 0) / max(n_attempted, 1)
    mean_rho = sum(m.rho for m in metrics if m.attempts > 0) / max(n_attempted, 1)

    # Per-state diagnostics (from feedback)
    drift_attempted = [m for m in metrics if m.topology_state == "drifting" and m.attempts > 0]
    drift_accepted = sum(1 for m in drift_attempted if m.accepted)
    drift_topo_improved = sum(1 for m in drift_attempted if m.topo_improved)
    collapse_attempted = [m for m in metrics if m.topology_state == "collapsing" and m.attempts > 0]
    collapse_accepted = sum(1 for m in collapse_attempted if m.accepted)
    collapse_topo_improved = sum(1 for m in collapse_attempted if m.topo_improved)

    return {
        "text": generated_text,
        "time": elapsed,
        "n_chunks": len(metrics),
        "n_accepted": n_accepted,
        "n_attempted": n_attempted,
        "n_stable": n_stable,
        "n_drifting": n_drifting,
        "n_collapsing": n_collapsing,
        "mean_kl": mean_kl,
        "mean_rho": mean_rho,
        # Per-state breakdown
        "drift_attempted": len(drift_attempted),
        "drift_accepted": drift_accepted,
        "drift_topo_improved": drift_topo_improved,
        "collapse_attempted": len(collapse_attempted),
        "collapse_accepted": collapse_accepted,
        "collapse_topo_improved": collapse_topo_improved,
        "metrics": [asdict(m) for m in metrics],
    }


def compute_self_consistency(samples: list[str]) -> float:
    """Jaccard similarity between pairs of samples."""
    if len(samples) < 2:
        return 1.0
    n_pairs = 0
    total = 0.0
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
                total += intersection / union
    return total / n_pairs if n_pairs > 0 else 1.0


def main():
    parser = argparse.ArgumentParser(description="LiveLoRA-Delta comparison experiment")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--n-samples", type=int, default=3, help="Samples for self-consistency")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="outputs/delta_comparison.json")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Model: {args.model} | Device: {device} | Prompts: {args.num_prompts}")
    print(f"Chunk size: {args.chunk_size} | Max tokens: {args.max_tokens}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    base_model = base_model.to(device)
    base_model.eval()

    # Wrap with LoRA once — reused across all methods and prompts
    lora_model = create_lora_model(base_model)

    prompts = PROMPTS[:args.num_prompts]
    methods = ["baseline", "ph", "entropy", "hybrid"]
    all_results = []

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {prompt[:60]}...")
        result = {"prompt": prompt}

        # Baseline (multiple samples for consistency)
        baseline_samples = []
        for s in range(args.n_samples):
            do_sample = s > 0  # First is greedy, rest sampled
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                gen = lora_model.model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    do_sample=do_sample,
                    temperature=0.7 if do_sample else 1.0,
                    top_p=0.9 if do_sample else 1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )
            text = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            baseline_samples.append(text)

        result["baseline"] = {
            "text": baseline_samples[0],
            "consistency": compute_self_consistency(baseline_samples),
        }

        # Delta methods
        for mode in ["ph", "entropy", "hybrid"]:
            print(f"  {mode}...", end=" ", flush=True)
            t0 = time.perf_counter()
            delta_result = run_delta_method(
                lora_model, tokenizer, prompt,
                optimization_mode=mode,
                max_new_tokens=args.max_tokens,
                chunk_size=args.chunk_size,
                device=device,
            )

            # Additional samples for self-consistency
            samples = [delta_result["text"]]
            for s in range(args.n_samples - 1):
                extra = run_delta_method(
                    lora_model, tokenizer, prompt,
                    optimization_mode=mode,
                    max_new_tokens=args.max_tokens,
                    chunk_size=args.chunk_size,
                    device=device,
                )
                samples.append(extra["text"])
            delta_result["consistency"] = compute_self_consistency(samples)

            result[mode] = delta_result
            elapsed = time.perf_counter() - t0
            print(f"ok ({elapsed:.1f}s) "
                  f"acc={delta_result['n_accepted']}/{delta_result['n_attempted']} "
                  f"drift={delta_result['n_drifting']} "
                  f"collapse={delta_result['n_collapsing']} "
                  f"cons={delta_result['consistency']:.3f}")

        print(f"  baseline consistency={result['baseline']['consistency']:.3f}")
        all_results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("LIVELORA-DELTA COMPARISON")
    print("=" * 70)

    for method in methods:
        if method == "baseline":
            consistencies = [r["baseline"]["consistency"] for r in all_results]
        else:
            consistencies = [r[method]["consistency"] for r in all_results]
        mean_c = sum(consistencies) / len(consistencies)
        print(f"\n{method:>12s}: mean_consistency={mean_c:.3f}")

        if method != "baseline":
            updates = [r[method]["n_accepted"] for r in all_results]
            attempted = [r[method]["n_attempted"] for r in all_results]
            drifts = [r[method]["n_drifting"] for r in all_results]
            collapses = [r[method]["n_collapsing"] for r in all_results]
            print(f"{'':>12s}  updates={sum(updates)}/{sum(attempted)} "
                  f"mean_drift_chunks={sum(drifts)/len(drifts):.1f} "
                  f"mean_collapse_chunks={sum(collapses)/len(collapses):.1f}")

    # Pairwise wins
    print("\n--- Pairwise wins (consistency) ---")
    for m1 in methods:
        for m2 in methods:
            if m1 >= m2:
                continue
            wins_1 = 0
            wins_2 = 0
            for r in all_results:
                c1 = r["baseline"]["consistency"] if m1 == "baseline" else r[m1]["consistency"]
                c2 = r["baseline"]["consistency"] if m2 == "baseline" else r[m2]["consistency"]
                if c1 > c2:
                    wins_1 += 1
                elif c2 > c1:
                    wins_2 += 1
            print(f"  {m1} vs {m2}: {wins_1}-{wins_2}")

    # Topology-triggered update statistics
    print("\n--- Update statistics ---")
    for mode in ["ph", "entropy", "hybrid"]:
        total_accepted = sum(r[mode]["n_accepted"] for r in all_results)
        total_attempted = sum(r[mode]["n_attempted"] for r in all_results)
        total_stable = sum(r[mode]["n_stable"] for r in all_results)
        total_drifting = sum(r[mode]["n_drifting"] for r in all_results)
        total_collapsing = sum(r[mode]["n_collapsing"] for r in all_results)
        total_chunks = sum(r[mode]["n_chunks"] for r in all_results)
        mean_kls = [r[mode]["mean_kl"] for r in all_results if r[mode]["n_attempted"] > 0]
        mean_rhos = [r[mode]["mean_rho"] for r in all_results if r[mode]["n_attempted"] > 0]

        print(f"\n  {mode}:")
        print(f"    chunks: {total_chunks}  stable={total_stable}  "
              f"drifting={total_drifting}  collapsing={total_collapsing}")
        print(f"    updates: {total_accepted}/{total_attempted} accepted "
              f"({100*total_accepted/max(total_attempted,1):.0f}%)")
        if mean_kls:
            print(f"    mean KL: {sum(mean_kls)/len(mean_kls):.6f}  "
                  f"mean rho: {sum(mean_rhos)/len(mean_rhos):.4f}")

        # Per-state diagnostics (from feedback: acceptance rate conditional on state)
        drift_att = sum(r[mode]["drift_attempted"] for r in all_results)
        drift_acc = sum(r[mode]["drift_accepted"] for r in all_results)
        drift_imp = sum(r[mode]["drift_topo_improved"] for r in all_results)
        collapse_att = sum(r[mode]["collapse_attempted"] for r in all_results)
        collapse_acc = sum(r[mode]["collapse_accepted"] for r in all_results)
        collapse_imp = sum(r[mode]["collapse_topo_improved"] for r in all_results)

        if drift_att > 0:
            print(f"    DRIFTING:   {drift_acc}/{drift_att} accepted ({100*drift_acc/drift_att:.0f}%), "
                  f"topo improved: {drift_imp}/{drift_att} ({100*drift_imp/drift_att:.0f}%)")
        if collapse_att > 0:
            print(f"    COLLAPSING: {collapse_acc}/{collapse_att} accepted ({100*collapse_acc/collapse_att:.0f}%), "
                  f"topo improved: {collapse_imp}/{collapse_att} ({100*collapse_imp/collapse_att:.0f}%)")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
