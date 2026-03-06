"""Gate-only ablation: isolate whether PH-Delta's win comes from the loss, the gate, or sparsity.

The critical experiment to understand the mechanism:

  1. PH-gradient + PH-gate    (full PH-Delta — existing winner)
  2. Entropy-grad + PH-gate   (entropy optimizer, PH structural accept/reject)
  3. Random noise + PH-gate   (no optimizer, just PH filtering random perturbations)
  4. Entropy-grad + accept-all (no gate, accept every update — existing entropy-delta)
  5. Entropy-grad + budget-matched (accept top-k by rho, same count as PH-Delta)

If (2) matches (1), the PH gate is the hero, not the PH gradient.
If (3) matches (1), even random noise filtered by PH works — pure gating.
If (5) matches (1), it's just about fewer updates, not the gate quality.

Run:
    python experiments/gate_ablation.py --model gpt2
    python experiments/gate_ablation.py --model Qwen/Qwen3.5-0.8B --device auto
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
    target_modules = get_lora_target_modules(base_model)
    lora_config = LiveLoraConfig(
        rank=8, alpha=16.0, target_modules=target_modules, dropout=0.0, bias="none",
    )
    lora_model = LiveLoraModel(base_model, lora_config)
    lora_model.freeze_base()
    return lora_model


def make_config(mode: str, chunk_size: int, max_tokens: int) -> DeltaConfig:
    """Create DeltaConfig for a given ablation mode."""
    # All modes use PH structural gate except "entropy" which uses topo-divergence gate
    if mode in ("ph", "entropy_ph_gate", "random"):
        tau_rho = 10.0
        epsilon_kl = 0.01
    else:
        tau_rho = 0.0
        epsilon_kl = 0.01

    return DeltaConfig(
        chunk_size=chunk_size,
        max_new_tokens=max_tokens,
        target_layers=[-1],
        max_points=64,
        max_dimension=0,
        lr=1e-4,
        epsilon_kl=epsilon_kl,
        tau_rho=tau_rho,
        cooldown_chunks=1,
        max_updates=5,
        conditional_ph=True,
        optimization_mode=mode,
        alpha_entropy=1.0,
        beta_topo=0.01,
        entropy_probe_len=8,
        divergence_drift_threshold=1.5,
    )


def run_method(lora_model, tokenizer, prompt, mode, chunk_size, max_tokens, device):
    """Run one prompt through a single ablation method."""
    config = make_config(mode, chunk_size, max_tokens)
    controller = GenerationController(lora_model, config)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    output_ids, metrics = controller.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        pad_token_id=tokenizer.pad_token_id,
    )
    elapsed = time.perf_counter() - t0

    text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    n_accepted = sum(1 for m in metrics if m.accepted)
    n_attempted = sum(1 for m in metrics if m.attempts > 0)
    n_drifting = sum(1 for m in metrics if m.topology_state == "drifting")
    topo_improved = sum(1 for m in metrics if m.attempts > 0 and m.topo_improved)

    return {
        "text": text,
        "time": elapsed,
        "n_chunks": len(metrics),
        "n_accepted": n_accepted,
        "n_attempted": n_attempted,
        "n_drifting": n_drifting,
        "topo_improved": topo_improved,
        "metrics": [asdict(m) for m in metrics],
    }


def compute_self_consistency(samples: list[str]) -> float:
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
    parser = argparse.ArgumentParser(description="Gate-only ablation experiment")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="outputs/gate_ablation.json")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Model: {args.model} | Device: {device} | Prompts: {args.num_prompts}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    lora_model = create_lora_model(base_model)

    # Ablation methods
    ablation_modes = [
        ("ph", "PH-grad + PH-gate"),
        ("entropy_ph_gate", "Entropy-grad + PH-gate"),
        ("random", "Random + PH-gate"),
        ("entropy", "Entropy-grad + topo-gate"),
    ]

    prompts = PROMPTS[:args.num_prompts]
    all_results = []

    # First pass: run all methods, collect PH-Delta acceptance counts per prompt
    # (needed for budget-matched entropy)
    ph_accepted_per_prompt = []

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {prompt[:60]}...")
        result = {"prompt": prompt}

        # Baseline
        baseline_samples = []
        for s in range(args.n_samples):
            do_sample = s > 0
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                gen = lora_model.model.generate(
                    **inputs, max_new_tokens=args.max_tokens,
                    do_sample=do_sample, temperature=0.7 if do_sample else 1.0,
                    top_p=0.9 if do_sample else 1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )
            text = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            baseline_samples.append(text)
        result["baseline"] = {"consistency": compute_self_consistency(baseline_samples)}

        # Run each ablation method
        for mode, label in ablation_modes:
            print(f"  {label}...", end=" ", flush=True)
            t0 = time.perf_counter()

            # Multiple samples for consistency
            samples = []
            first_result = None
            for s in range(args.n_samples):
                r = run_method(
                    lora_model, tokenizer, prompt, mode,
                    args.chunk_size, args.max_tokens, device,
                )
                samples.append(r["text"])
                if first_result is None:
                    first_result = r

            first_result["consistency"] = compute_self_consistency(samples)
            result[mode] = first_result
            elapsed = time.perf_counter() - t0
            print(f"acc={first_result['n_accepted']}/{first_result['n_attempted']} "
                  f"cons={first_result['consistency']:.3f} ({elapsed:.1f}s)")

            if mode == "ph":
                ph_accepted_per_prompt.append(first_result["n_accepted"])

        all_results.append(result)

    # Budget-matched entropy: re-run entropy with max_updates = mean PH acceptance
    mean_ph_accepted = sum(ph_accepted_per_prompt) / max(len(ph_accepted_per_prompt), 1)
    budget = max(1, round(mean_ph_accepted))
    print(f"\n--- Budget-matched entropy (max_updates={budget}, matching PH mean={mean_ph_accepted:.1f}) ---")

    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] ", end="", flush=True)
        config = make_config("entropy", args.chunk_size, args.max_tokens)
        config.max_updates = budget

        samples = []
        first_result = None
        for s in range(args.n_samples):
            controller = GenerationController(lora_model, config)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            output_ids, metrics = controller.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                pad_token_id=tokenizer.pad_token_id,
            )
            text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            samples.append(text)
            if first_result is None:
                n_acc = sum(1 for m in metrics if m.accepted)
                n_att = sum(1 for m in metrics if m.attempts > 0)
                first_result = {
                    "n_accepted": n_acc, "n_attempted": n_att,
                    "n_chunks": len(metrics),
                    "metrics": [asdict(m) for m in metrics],
                }

        first_result["text"] = samples[0]
        first_result["consistency"] = compute_self_consistency(samples)
        all_results[i]["entropy_budgeted"] = first_result
        print(f"acc={first_result['n_accepted']}/{first_result['n_attempted']} "
              f"cons={first_result['consistency']:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("GATE ABLATION RESULTS")
    print("=" * 70)

    all_modes = [("baseline", "Baseline")] + ablation_modes + [("entropy_budgeted", "Entropy-budgeted")]

    for mode_key, label in all_modes:
        if mode_key == "baseline":
            consistencies = [r["baseline"]["consistency"] for r in all_results]
        else:
            consistencies = [r[mode_key]["consistency"] for r in all_results]
        mean_c = sum(consistencies) / len(consistencies)

        if mode_key == "baseline":
            print(f"\n  {label:30s}: consistency={mean_c:.3f}")
        else:
            updates = [r[mode_key]["n_accepted"] for r in all_results]
            attempted = [r[mode_key]["n_attempted"] for r in all_results]
            print(f"  {label:30s}: consistency={mean_c:.3f}  "
                  f"updates={sum(updates)}/{sum(attempted)}")

    # Pairwise wins vs PH-Delta
    print("\n--- vs PH-Delta (pairwise wins) ---")
    for mode_key, label in all_modes:
        if mode_key == "ph":
            continue
        ph_wins = 0
        other_wins = 0
        for r in all_results:
            c_ph = r["ph"]["consistency"]
            c_other = r["baseline"]["consistency"] if mode_key == "baseline" else r[mode_key]["consistency"]
            if c_ph > c_other:
                ph_wins += 1
            elif c_other > c_ph:
                other_wins += 1
        print(f"  PH-Delta vs {label:25s}: {ph_wins}-{other_wins}")

    # The key question: where does the win come from?
    print("\n--- Interpretation ---")
    ph_cons = sum(r["ph"]["consistency"] for r in all_results) / len(all_results)
    ent_ph_cons = sum(r["entropy_ph_gate"]["consistency"] for r in all_results) / len(all_results)
    rand_cons = sum(r["random"]["consistency"] for r in all_results) / len(all_results)
    ent_cons = sum(r["entropy"]["consistency"] for r in all_results) / len(all_results)
    budget_cons = sum(r["entropy_budgeted"]["consistency"] for r in all_results) / len(all_results)

    if abs(ent_ph_cons - ph_cons) < 0.05:
        print("  -> Entropy+PH-gate ~ PH-Delta: the PH GATE is the hero, not the PH gradient")
    elif ent_ph_cons > ph_cons:
        print("  -> Entropy+PH-gate > PH-Delta: entropy gradient is better, PH gate helps both")
    else:
        print("  -> PH-Delta > Entropy+PH-gate: the PH gradient matters, not just the gate")

    if abs(rand_cons - ph_cons) < 0.05:
        print("  -> Random+PH-gate ~ PH-Delta: pure gating works, even random perturbations suffice")
    else:
        print(f"  -> Random+PH-gate ({rand_cons:.3f}) != PH-Delta ({ph_cons:.3f}): optimization matters")

    if abs(budget_cons - ph_cons) < 0.05:
        print("  -> Budget-matched entropy ~ PH-Delta: it's about fewer updates, not gate quality")
    else:
        print(f"  -> Budget-matched ({budget_cons:.3f}) != PH-Delta ({ph_cons:.3f}): gate quality matters")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
