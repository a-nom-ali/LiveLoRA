"""Acceptance threshold sweep: validate the MCMC interpretation.

Vary tau_rho across a range to produce acceptance rates from ~10% to ~80%,
testing whether performance peaks near 20-30% acceptance (as predicted by
the stochastic search / Metropolis interpretation).

Uses entropy_ph_gate mode (the best-performing configuration) and sweeps
tau_rho values to modulate how strict the PH structural gate is.

Run:
    python experiments/threshold_sweep.py --model gpt2
    python experiments/threshold_sweep.py --model Qwen/Qwen3.5-0.8B --device auto
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


def create_lora_model(base_model):
    target_modules = get_lora_target_modules(base_model)
    lora_config = LiveLoraConfig(
        rank=8, alpha=16.0, target_modules=target_modules, dropout=0.0, bias="none",
    )
    lora_model = LiveLoraModel(base_model, lora_config)
    lora_model.freeze_base()
    return lora_model


def run_sweep_point(lora_model, tokenizer, prompts, tau_rho, n_samples, chunk_size, max_tokens, device):
    """Run all prompts at a single tau_rho value."""
    results = []

    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] ", end="", flush=True)
        config = DeltaConfig(
            chunk_size=chunk_size,
            max_new_tokens=max_tokens,
            target_layers=[-1],
            max_points=64,
            max_dimension=0,
            lr=1e-4,
            epsilon_kl=0.01,
            tau_rho=tau_rho,
            cooldown_chunks=1,
            max_updates=5,
            conditional_ph=True,
            optimization_mode="entropy_ph_gate",
            alpha_entropy=1.0,
            beta_topo=0.01,
            entropy_probe_len=8,
            divergence_drift_threshold=1.5,
        )

        samples = []
        first_result = None
        for s in range(n_samples):
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
                first_result = {
                    "n_accepted": sum(1 for m in metrics if m.accepted),
                    "n_attempted": sum(1 for m in metrics if m.attempts > 0),
                    "n_chunks": len(metrics),
                }

        first_result["consistency"] = compute_self_consistency(samples)
        results.append(first_result)
        acc_rate = first_result["n_accepted"] / max(first_result["n_attempted"], 1) * 100
        print(f"cons={first_result['consistency']:.3f} acc={acc_rate:.0f}%", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Acceptance threshold sweep")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="outputs/threshold_sweep.json")
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

    prompts = PROMPTS[:args.num_prompts]

    # Sweep tau_rho: higher = stricter gate = fewer acceptances
    # 0.0 = accept everything that improves structure at all
    # Very high = only accept very efficient updates
    tau_rho_values = [0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]

    all_sweep_results = {}

    for tau in tau_rho_values:
        print(f"\n{'='*60}")
        print(f"tau_rho = {tau}")
        print(f"{'='*60}")

        t0 = time.perf_counter()
        results = run_sweep_point(
            lora_model, tokenizer, prompts, tau,
            args.n_samples, args.chunk_size, args.max_tokens, device,
        )
        elapsed = time.perf_counter() - t0

        consistencies = [r["consistency"] for r in results]
        mean_c = sum(consistencies) / len(consistencies)
        total_accepted = sum(r["n_accepted"] for r in results)
        total_attempted = sum(r["n_attempted"] for r in results)
        acceptance_rate = total_accepted / max(total_attempted, 1) * 100

        print(f"  consistency={mean_c:.3f}  acceptance={acceptance_rate:.0f}% "
              f"({total_accepted}/{total_attempted})  time={elapsed:.0f}s")

        all_sweep_results[str(tau)] = {
            "tau_rho": tau,
            "mean_consistency": mean_c,
            "acceptance_rate": acceptance_rate,
            "total_accepted": total_accepted,
            "total_attempted": total_attempted,
            "per_prompt": results,
            "elapsed": elapsed,
        }

    # Summary table
    print(f"\n{'='*60}")
    print("THRESHOLD SWEEP RESULTS")
    print(f"{'='*60}")
    print(f"  {'tau_rho':>10}  {'Consistency':>12}  {'Acceptance':>12}  {'Updates':>10}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*10}")
    for tau_str, data in sorted(all_sweep_results.items(), key=lambda x: x[1]["tau_rho"]):
        print(f"  {data['tau_rho']:>10.1f}  {data['mean_consistency']:>12.3f}  "
              f"{data['acceptance_rate']:>11.0f}%  "
              f"{data['total_accepted']:>4}/{data['total_attempted']}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_sweep_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
