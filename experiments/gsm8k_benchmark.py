"""Ground truth benchmark: does PH-gated adaptation improve correctness?

Tests on GSM8K (grade school math) where we have verifiable answers.
This is the critical missing experiment — does high consistency = correctness?

Compares:
  1. Baseline (no adaptation)
  2. Entropy + PH-gate (best from gate ablation)
  3. Entropy + no gate (accept all)

Tracks:
  - accuracy (exact match on final number)
  - consistency (self-consistency across samples)
  - acceptance rate
  - topology improvement rate

Run:
    python experiments/gsm8k_benchmark.py --model Qwen/Qwen3.5-0.8B --device auto
    python experiments/gsm8k_benchmark.py --model gpt2 --num-problems 5  # quick test
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from livelora.core.gen_controller import DeltaConfig, GenerationController
from livelora.core.lora_adapter import LiveLoraConfig, LiveLoraModel, get_lora_target_modules


def load_gsm8k(split: str = "test", max_problems: int | None = None) -> list[dict]:
    """Load GSM8K from HuggingFace datasets."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split=split)
        problems = []
        for item in ds:
            # Extract the final numeric answer from the "#### NUMBER" format
            answer_text = item["answer"]
            match = re.search(r"####\s*(.+)", answer_text)
            if match:
                answer = match.group(1).strip().replace(",", "")
            else:
                answer = answer_text.strip()
            problems.append({
                "question": item["question"],
                "answer": answer,
                "full_solution": answer_text,
            })
            if max_problems and len(problems) >= max_problems:
                break
        return problems
    except ImportError:
        print("WARNING: 'datasets' not installed. Install with: pip install datasets")
        print("Falling back to built-in mini benchmark.")
        return _builtin_problems()[:max_problems] if max_problems else _builtin_problems()


def _builtin_problems() -> list[dict]:
    """Fallback mini-benchmark if datasets not installed."""
    return [
        {"question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?", "answer": "18"},
        {"question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", "answer": "3"},
        {"question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?", "answer": "70000"},
        {"question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?", "answer": "540"},
        {"question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. If Wendi has 20 chickens, how many cups of feed does she need to give her chickens in the final meal?", "answer": "20"},
        {"question": "Kylar went to the store to get his decor items. He bought 2 vases, 1 plate, and 1 pillow. The vase cost $20 each, the plate cost $15, and the pillow cost $10. How much did he spend in all?", "answer": "65"},
        {"question": "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?", "answer": "260"},
        {"question": "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How long does it take to download the file?", "answer": "160"},
        {"question": "John takes care of 10 dogs. Each dog takes .5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?", "answer": "35"},
        {"question": "Gail has two fish tanks. The first tank has 48 fish while the second tank has 32 fish. She buys 8 more fish for the first tank and 14 more for the second tank. How many fish does she have in total?", "answer": "102"},
    ]


def extract_number(text: str) -> str | None:
    """Extract the final number from generated text.

    Tries several patterns:
    1. "#### NUMBER" format (GSM8K style)
    2. "The answer is NUMBER"
    3. Last number in the text
    """
    # GSM8K format
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    # "The answer is X" pattern
    match = re.search(r"(?:the answer is|answer:?|=)\s*\$?([\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")

    # Last number in text (fallback)
    numbers = re.findall(r"(?<!\d)([\d,]+(?:\.\d+)?)(?!\d)", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def check_answer(generated: str, gold: str) -> bool:
    """Check if the generated answer matches the gold answer."""
    extracted = extract_number(generated)
    if extracted is None:
        return False
    # Normalize both
    try:
        return float(extracted) == float(gold)
    except ValueError:
        return extracted.strip() == gold.strip()


def format_prompt(question: str) -> str:
    """Format a GSM8K question as a prompt."""
    return (
        f"Solve the following math problem step by step. "
        f"Show your work and end with 'The answer is [NUMBER]'.\n\n"
        f"Question: {question}\n\n"
        f"Solution:"
    )


def create_lora_model(base_model):
    target_modules = get_lora_target_modules(base_model)
    lora_config = LiveLoraConfig(
        rank=8, alpha=16.0, target_modules=target_modules, dropout=0.0, bias="none",
    )
    lora_model = LiveLoraModel(base_model, lora_config)
    lora_model.freeze_base()
    return lora_model


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


def run_baseline(lora_model, tokenizer, prompt, n_samples, max_tokens, device):
    """Run baseline (no adaptation) generation."""
    samples = []
    for s in range(n_samples):
        do_sample = s > 0
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = lora_model.model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=do_sample, temperature=0.7 if do_sample else 1.0,
                top_p=0.9 if do_sample else 1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        samples.append(text)
    return samples


def run_delta(lora_model, tokenizer, prompt, mode, tau_rho, n_samples, max_tokens, chunk_size, device):
    """Run LiveLoRA-Delta generation with specified mode."""
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
        optimization_mode=mode,
        alpha_entropy=1.0,
        beta_topo=0.01,
        entropy_probe_len=8,
        divergence_drift_threshold=1.5,
    )

    samples = []
    all_metrics = []
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
        if s == 0:
            all_metrics = metrics
    return samples, all_metrics


def main():
    parser = argparse.ArgumentParser(description="GSM8K ground truth benchmark")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--num-problems", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="outputs/gsm8k_benchmark.json")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Model: {args.model} | Device: {device} | Problems: {args.num_problems}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    lora_model = create_lora_model(base_model)

    problems = load_gsm8k("test", args.num_problems)
    print(f"Loaded {len(problems)} problems")

    methods = [
        ("baseline", "Baseline (no adaptation)"),
        ("entropy_ph_gate", "Entropy + PH-gate (tau=10)"),
        ("entropy", "Entropy + topo-gate (no PH gate)"),
    ]

    all_results = []

    for i, problem in enumerate(problems):
        prompt = format_prompt(problem["question"])
        gold = problem["answer"]
        print(f"\n[{i+1}/{len(problems)}] {problem['question'][:60]}...")
        print(f"  Gold answer: {gold}")

        result = {"question": problem["question"], "gold_answer": gold}

        for mode, label in methods:
            t0 = time.perf_counter()

            if mode == "baseline":
                samples = run_baseline(
                    lora_model, tokenizer, prompt, args.n_samples, args.max_tokens, device,
                )
                metrics = []
            else:
                tau = 10.0 if mode == "entropy_ph_gate" else 0.0
                samples, metrics = run_delta(
                    lora_model, tokenizer, prompt, mode, tau,
                    args.n_samples, args.max_tokens, args.chunk_size, device,
                )

            elapsed = time.perf_counter() - t0

            # Check correctness of first (greedy) sample
            correct = check_answer(samples[0], gold)
            extracted = extract_number(samples[0])
            consistency = compute_self_consistency(samples)

            # Check majority vote correctness
            answers = [extract_number(s) for s in samples]
            majority_correct = False
            if answers:
                from collections import Counter
                answer_counts = Counter(a for a in answers if a is not None)
                if answer_counts:
                    majority_answer = answer_counts.most_common(1)[0][0]
                    try:
                        majority_correct = float(majority_answer) == float(gold)
                    except (ValueError, TypeError):
                        majority_correct = majority_answer == gold

            n_accepted = sum(1 for m in metrics if m.accepted) if metrics else 0
            n_attempted = sum(1 for m in metrics if m.attempts > 0) if metrics else 0
            topo_improved = sum(1 for m in metrics if m.topo_improved) if metrics else 0

            result[mode] = {
                "correct": correct,
                "majority_correct": majority_correct,
                "extracted_answer": extracted,
                "consistency": consistency,
                "n_accepted": n_accepted,
                "n_attempted": n_attempted,
                "topo_improved": topo_improved,
                "time": elapsed,
                "text_sample": samples[0][:500],
            }

            status = "CORRECT" if correct else "wrong"
            maj_status = "MAJ_OK" if majority_correct else "maj_no"
            acc_info = f"acc={n_accepted}/{n_attempted}" if mode != "baseline" else ""
            print(f"  {label:35s}: {status:7s} {maj_status:6s} "
                  f"ext={extracted or '?':>8s} cons={consistency:.3f} {acc_info} ({elapsed:.1f}s)")

        all_results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("GSM8K BENCHMARK RESULTS")
    print(f"{'='*70}")

    for mode, label in methods:
        correct = sum(1 for r in all_results if r[mode]["correct"])
        maj_correct = sum(1 for r in all_results if r[mode]["majority_correct"])
        consistencies = [r[mode]["consistency"] for r in all_results]
        mean_c = sum(consistencies) / len(consistencies)
        n = len(all_results)

        print(f"\n  {label}")
        print(f"    Accuracy (greedy):   {correct}/{n} ({correct/n*100:.1f}%)")
        print(f"    Accuracy (majority): {maj_correct}/{n} ({maj_correct/n*100:.1f}%)")
        print(f"    Consistency:         {mean_c:.3f}")
        if mode != "baseline":
            total_acc = sum(r[mode]["n_accepted"] for r in all_results)
            total_att = sum(r[mode]["n_attempted"] for r in all_results)
            total_topo = sum(r[mode]["topo_improved"] for r in all_results)
            print(f"    Updates:             {total_acc}/{total_att}")
            print(f"    Topo improved:       {total_topo}")

    # Pairwise comparison
    print(f"\n--- Entropy+PH-gate vs Baseline ---")
    gated_wins = sum(1 for r in all_results if r["entropy_ph_gate"]["correct"] and not r["baseline"]["correct"])
    baseline_wins = sum(1 for r in all_results if r["baseline"]["correct"] and not r["entropy_ph_gate"]["correct"])
    both_correct = sum(1 for r in all_results if r["entropy_ph_gate"]["correct"] and r["baseline"]["correct"])
    both_wrong = sum(1 for r in all_results if not r["entropy_ph_gate"]["correct"] and not r["baseline"]["correct"])
    print(f"  Gated wins: {gated_wins}  Baseline wins: {baseline_wins}  "
          f"Both correct: {both_correct}  Both wrong: {both_wrong}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
