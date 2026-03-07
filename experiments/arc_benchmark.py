"""ARC benchmark: does PH-gated adaptation improve correctness?

Tests on ARC-Challenge (multiple choice science questions) with chain-of-thought
prompting to generate longer reasoning chains where topology tracking can act.
ARC-Challenge baseline for small models is typically 30-55%.

Compares:
  1. Baseline (no adaptation)
  2. Entropy + PH-gate (best from gate ablation)
  3. Entropy + topo-gate (accept all)

Run:
    python experiments/arc_benchmark.py --model Qwen/Qwen3.5-0.8B --device auto
    python experiments/arc_benchmark.py --model Qwen/Qwen3.5-4B --quantize 4bit --device auto
    python experiments/arc_benchmark.py --model Qwen/Qwen3.5-9B --quantize 4bit --device auto
    python experiments/arc_benchmark.py --split easy --model Qwen/Qwen3.5-0.8B  # use ARC-Easy instead
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from livelora.core.gen_controller import DeltaConfig, GenerationController
from livelora.core.lora_adapter import LiveLoraConfig, LiveLoraModel, get_lora_target_modules


def load_arc(difficulty: str = "challenge", split: str = "test", max_problems: int | None = None) -> list[dict]:
    """Load ARC from HuggingFace datasets.

    Args:
        difficulty: "challenge" (harder, ~30-55% baseline) or "easy" (~60-80% baseline)
        split: dataset split
        max_problems: limit number of problems
    """
    config = "ARC-Challenge" if difficulty == "challenge" else "ARC-Easy"
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/ai2_arc", config, split=split)
        problems = []
        for item in ds:
            problems.append({
                "id": item["id"],
                "question": item["question"],
                "choices": item["choices"]["text"],
                "labels": item["choices"]["label"],
                "answer": item["answerKey"],
            })
            if max_problems and len(problems) >= max_problems:
                break
        return problems
    except ImportError:
        print("WARNING: 'datasets' not installed. Install with: pip install datasets")
        return _builtin_problems()[:max_problems] if max_problems else _builtin_problems()


def _builtin_problems() -> list[dict]:
    """Fallback mini-benchmark."""
    return [
        {"id": "1", "question": "Which statement best explains why photosynthesis is the foundation of most food webs?",
         "choices": ["Sunlight is the source of energy for nearly all ecosystems.", "Most ecosystems are found on land instead of in water.", "Carbon dioxide is more available than other gases.", "The producers in all ecosystems are plants."],
         "labels": ["A", "B", "C", "D"], "answer": "A"},
        {"id": "2", "question": "What is the boiling point of water at sea level?",
         "choices": ["50°C", "100°C", "150°C", "200°C"],
         "labels": ["A", "B", "C", "D"], "answer": "B"},
        {"id": "3", "question": "Which planet is closest to the Sun?",
         "choices": ["Venus", "Earth", "Mercury", "Mars"],
         "labels": ["A", "B", "C", "D"], "answer": "C"},
    ]


def format_prompt(question: str, choices: list[str], labels: list[str]) -> str:
    """Format an ARC question requiring chain-of-thought reasoning."""
    choice_text = "\n".join(f"{label}. {text}" for label, text in zip(labels, choices))
    return (
        f"Question: {question}\n{choice_text}\n\n"
        f"Think through this step by step, then give your final answer as "
        f"'The answer is X' where X is A, B, C, or D.\n\n"
        f"Reasoning:"
    )


def extract_answer(text: str, valid_labels: list[str]) -> str | None:
    """Extract the answer letter from generated text."""
    text = text.strip()
    # First non-whitespace character if it's a valid label
    if text and text[0].upper() in valid_labels:
        return text[0].upper()
    # "The answer is X" pattern
    match = re.search(r"(?:answer is|answer:)\s*([A-D])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # First occurrence of a standalone label letter
    match = re.search(r"\b([A-D])\b", text)
    if match:
        return match.group(1).upper()
    return None


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


def compute_answer_agreement(answers: list[str | None]) -> float:
    """Fraction of sample pairs that agree on the answer letter."""
    if len(answers) < 2:
        return 1.0
    n_pairs = 0
    agree = 0
    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            n_pairs += 1
            if answers[i] is not None and answers[i] == answers[j]:
                agree += 1
    return agree / n_pairs if n_pairs > 0 else 0.0


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
    parser = argparse.ArgumentParser(description="ARC-Easy benchmark")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--num-problems", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quantize", choices=["none", "4bit", "8bit"], default="none",
                        help="Load model with bitsandbytes quantization (4bit/8bit)")
    parser.add_argument("--split", choices=["challenge", "easy"], default="challenge",
                        help="ARC difficulty: challenge (harder) or easy")
    parser.add_argument("--output", default="outputs/arc_benchmark.json")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    quant_label = f" | Quantize: {args.quantize}" if args.quantize != "none" else ""
    print(f"Model: {args.model} | Device: {device} | Problems: {args.num_problems}{quant_label}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.quantize != "none":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(args.quantize == "4bit"),
            load_in_8bit=(args.quantize == "8bit"),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb_config, device_map="auto",
        ).eval()
        from peft import prepare_model_for_kbit_training
        base_model = prepare_model_for_kbit_training(base_model)
        device = next(base_model.parameters()).device
    else:
        base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()

    lora_model = create_lora_model(base_model)

    problems = load_arc(args.split, "test", args.num_problems)
    print(f"Loaded {len(problems)} ARC-{args.split.title()} problems", flush=True)

    methods = [
        ("baseline", "Baseline (no adaptation)"),
        ("entropy_ph_gate", "Entropy + PH-gate (tau=10)"),
        ("entropy", "Entropy + topo-gate (no PH gate)"),
    ]

    all_results = []

    for i, problem in enumerate(problems):
        prompt = format_prompt(problem["question"], problem["choices"], problem["labels"])
        gold = problem["answer"]
        print(f"\n[{i+1}/{len(problems)}] {problem['question'][:70]}...", flush=True)
        print(f"  Gold: {gold}", flush=True)

        result = {"id": problem["id"], "question": problem["question"], "gold_answer": gold}

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

            # Extract answers from all samples
            answers = [extract_answer(s, problem["labels"]) for s in samples]
            correct = answers[0] == gold if answers[0] else False

            # Majority vote
            from collections import Counter
            answer_counts = Counter(a for a in answers if a is not None)
            majority_answer = answer_counts.most_common(1)[0][0] if answer_counts else None
            majority_correct = majority_answer == gold if majority_answer else False

            consistency = compute_self_consistency(samples)
            agreement = compute_answer_agreement(answers)

            n_accepted = sum(1 for m in metrics if m.accepted) if metrics else 0
            n_attempted = sum(1 for m in metrics if m.attempts > 0) if metrics else 0

            result[mode] = {
                "correct": correct,
                "majority_correct": majority_correct,
                "extracted_answer": answers[0],
                "all_answers": answers,
                "agreement": agreement,
                "consistency": consistency,
                "n_accepted": n_accepted,
                "n_attempted": n_attempted,
                "time": elapsed,
                "text_sample": samples[0][:200],
            }

            status = "CORRECT" if correct else "wrong"
            maj_status = "MAJ_OK" if majority_correct else "maj_no"
            acc_info = f"acc={n_accepted}/{n_attempted}" if mode != "baseline" else ""
            print(f"  {label:35s}: {status:7s} {maj_status:6s} "
                  f"ext={answers[0] or '?':>2s} agr={agreement:.2f} {acc_info} ({elapsed:.1f}s)", flush=True)

        all_results.append(result)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print(f"ARC-{args.split.upper()} BENCHMARK RESULTS", flush=True)
    print(f"{'='*70}", flush=True)

    for mode, label in methods:
        correct = sum(1 for r in all_results if r[mode]["correct"])
        maj_correct = sum(1 for r in all_results if r[mode]["majority_correct"])
        consistencies = [r[mode]["consistency"] for r in all_results]
        agreements = [r[mode]["agreement"] for r in all_results]
        mean_c = sum(consistencies) / len(consistencies)
        mean_a = sum(agreements) / len(agreements)
        n = len(all_results)

        print(f"\n  {label}", flush=True)
        print(f"    Accuracy (greedy):   {correct}/{n} ({correct/n*100:.1f}%)", flush=True)
        print(f"    Accuracy (majority): {maj_correct}/{n} ({maj_correct/n*100:.1f}%)", flush=True)
        print(f"    Consistency:         {mean_c:.3f}", flush=True)
        print(f"    Answer agreement:    {mean_a:.3f}", flush=True)
        if mode != "baseline":
            total_acc = sum(r[mode]["n_accepted"] for r in all_results)
            total_att = sum(r[mode]["n_attempted"] for r in all_results)
            print(f"    Updates:             {total_acc}/{total_att}", flush=True)

    # Pairwise comparison
    print(f"\n--- Entropy+PH-gate vs Baseline ---", flush=True)
    gated_wins = sum(1 for r in all_results if r["entropy_ph_gate"]["correct"] and not r["baseline"]["correct"])
    baseline_wins = sum(1 for r in all_results if r["baseline"]["correct"] and not r["entropy_ph_gate"]["correct"])
    both_correct = sum(1 for r in all_results if r["entropy_ph_gate"]["correct"] and r["baseline"]["correct"])
    both_wrong = sum(1 for r in all_results if not r["entropy_ph_gate"]["correct"] and not r["baseline"]["correct"])
    print(f"  Gated wins: {gated_wins}  Baseline wins: {baseline_wins}  "
          f"Both correct: {both_correct}  Both wrong: {both_wrong}", flush=True)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
