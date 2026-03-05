"""Correlation study: does activation topology predict output quality?

The critical question before Phase 1: if topology doesn't correlate
with quality, PH may optimize something real but not aligned with
what we care about.

For each prompt:
1. Run baseline generation (no adaptation) in chunks
2. Compute topology at each chunk (Betti numbers, persistence, tracker state)
3. Measure output quality via self-consistency (agreement across N samples)
4. Correlate: collapse frequency <-> error rate, divergence <-> error rate

Run:
    python experiments/correlation_study.py --model gpt2 --num-prompts 20
    python experiments/correlation_study.py --model Qwen/Qwen3.5-0.8B --num-prompts 50
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from livelora.topology.ph_tracker import PHTracker, TopologyState


# Simple reasoning prompts for correlation testing
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
class ChunkTopology:
    """Topology observation for a single generation chunk."""

    chunk_idx: int
    state: str  # STABLE, DRIFTING, COLLAPSING
    betti_0: int
    betti_1: int
    total_persistence: float
    divergence_from_baseline: float


@dataclass
class PromptResult:
    """Full result for one prompt across multiple samples."""

    prompt: str
    chunk_topologies: list[ChunkTopology]
    samples: list[str]
    self_consistency: float  # Agreement across samples (0-1)
    collapse_count: int  # Number of COLLAPSING chunks
    drift_count: int  # Number of DRIFTING chunks
    mean_divergence: float  # Average divergence from baseline
    first_instability_chunk: int  # First non-STABLE chunk (-1 if all stable)
    generation_time: float


def compute_self_consistency(samples: list[str]) -> float:
    """Simple self-consistency: fraction of sample pairs that agree.

    Uses normalized text overlap as a proxy for agreement.
    """
    if len(samples) < 2:
        return 1.0

    n_pairs = 0
    agreements = 0

    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            n_pairs += 1
            # Normalize: lowercase, strip whitespace, split into words
            words_i = set(samples[i].lower().split())
            words_j = set(samples[j].lower().split())

            if not words_i or not words_j:
                continue

            # Jaccard similarity
            intersection = len(words_i & words_j)
            union = len(words_i | words_j)
            if union > 0:
                agreements += intersection / union

    return agreements / n_pairs if n_pairs > 0 else 1.0


def run_with_topology(
    model,
    tokenizer,
    prompt: str,
    tracker: PHTracker,
    chunk_size: int = 32,
    max_new_tokens: int = 128,
    device: str = "cpu",
    target_layer: int = -1,
) -> tuple[str, list[ChunkTopology]]:
    """Generate text while tracking topology per chunk."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    current_ids = inputs["input_ids"]
    current_mask = inputs.get("attention_mask")

    # Get baseline topology from prompt
    with torch.no_grad():
        outputs = model(
            input_ids=current_ids,
            attention_mask=current_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states
        n_layers = len(hidden_states)
        layer_idx = n_layers + target_layer if target_layer < 0 else target_layer
        baseline_acts = hidden_states[layer_idx][0]  # (seq, dim)
        tracker.reset()
        tracker.set_baseline(baseline_acts)

    chunk_topos = []
    total_generated = 0

    while total_generated < max_new_tokens:
        tokens_to_gen = min(chunk_size, max_new_tokens - total_generated)

        with torch.no_grad():
            gen_output = model.generate(
                input_ids=current_ids,
                attention_mask=current_mask,
                max_new_tokens=tokens_to_gen,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = gen_output.shape[1] - current_ids.shape[1]
        if new_tokens == 0:
            break

        current_ids = gen_output
        if current_mask is not None:
            current_mask = torch.cat(
                [current_mask, torch.ones(1, new_tokens, device=device, dtype=current_mask.dtype)],
                dim=1,
            )
        total_generated += new_tokens

        # Compute topology on this chunk's activations
        with torch.no_grad():
            outputs = model(
                input_ids=current_ids,
                attention_mask=current_mask,
                output_hidden_states=True,
            )
            acts = outputs.hidden_states[layer_idx][0]
            summary = tracker.observe(acts)
            state = tracker.assess()

        chunk_topos.append(ChunkTopology(
            chunk_idx=len(chunk_topos),
            state=state.value,
            betti_0=summary.betti_0,
            betti_1=summary.betti_1,
            total_persistence=summary.total_persistence,
            divergence_from_baseline=tracker.divergence_from_baseline(),
        ))

    # Decode
    generated_text = tokenizer.decode(
        current_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return generated_text, chunk_topos


def run_prompt(
    model,
    tokenizer,
    prompt: str,
    n_samples: int = 3,
    chunk_size: int = 32,
    max_new_tokens: int = 128,
    device: str = "cpu",
) -> PromptResult:
    """Run one prompt with topology tracking and self-consistency."""
    tracker = PHTracker(max_points=64, max_dimension=0)

    t0 = time.perf_counter()

    # First run with topology tracking (deterministic)
    text, chunk_topos = run_with_topology(
        model, tokenizer, prompt, tracker,
        chunk_size=chunk_size, max_new_tokens=max_new_tokens, device=device,
    )

    # Additional samples for self-consistency (with sampling)
    samples = [text]
    for _ in range(n_samples - 1):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        sample_text = tokenizer.decode(
            gen[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        samples.append(sample_text)

    elapsed = time.perf_counter() - t0

    # Compute aggregate topology stats
    collapse_count = sum(1 for c in chunk_topos if c.state == "collapsing")
    drift_count = sum(1 for c in chunk_topos if c.state == "drifting")
    divergences = [c.divergence_from_baseline for c in chunk_topos]
    mean_div = sum(divergences) / len(divergences) if divergences else 0.0

    first_instability = -1
    for c in chunk_topos:
        if c.state != "stable":
            first_instability = c.chunk_idx
            break

    return PromptResult(
        prompt=prompt,
        chunk_topologies=chunk_topos,
        samples=samples,
        self_consistency=compute_self_consistency(samples),
        collapse_count=collapse_count,
        drift_count=drift_count,
        mean_divergence=mean_div,
        first_instability_chunk=first_instability,
        generation_time=elapsed,
    )


def main():
    parser = argparse.ArgumentParser(description="Topology-quality correlation study")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name")
    parser.add_argument("--num-prompts", type=int, default=20, help="Number of prompts to test")
    parser.add_argument("--n-samples", type=int, default=3, help="Samples per prompt for consistency")
    parser.add_argument("--chunk-size", type=int, default=32, help="Tokens per chunk")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max generation length")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="outputs/correlation_study.json", help="Output file")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Model: {args.model} | Device: {device} | Prompts: {args.num_prompts}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    if device == "cpu":
        model = model.to(device)
    model.eval()

    prompts = DEFAULT_PROMPTS[:args.num_prompts]
    results = []

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {prompt[:60]}...")
        result = run_prompt(
            model, tokenizer, prompt,
            n_samples=args.n_samples,
            chunk_size=args.chunk_size,
            max_new_tokens=args.max_tokens,
            device=device,
        )
        results.append(result)
        print(f"  consistency={result.self_consistency:.2f}  "
              f"collapses={result.collapse_count}  "
              f"drifts={result.drift_count}  "
              f"mean_div={result.mean_divergence:.3f}  "
              f"time={result.generation_time:.1f}s")

    # Summary statistics
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    consistencies = [r.self_consistency for r in results]
    collapses = [r.collapse_count for r in results]
    drifts = [r.drift_count for r in results]
    divergences = [r.mean_divergence for r in results]

    print(f"\nSelf-consistency:  mean={sum(consistencies)/len(consistencies):.3f}  "
          f"min={min(consistencies):.3f}  max={max(consistencies):.3f}")
    print(f"Collapse chunks:   mean={sum(collapses)/len(collapses):.1f}  "
          f"max={max(collapses)}")
    print(f"Drifting chunks:   mean={sum(drifts)/len(drifts):.1f}  "
          f"max={max(drifts)}")
    print(f"Mean divergence:   mean={sum(divergences)/len(divergences):.4f}  "
          f"max={max(divergences):.4f}")

    # Split into high/low consistency groups
    median_consistency = sorted(consistencies)[len(consistencies) // 2]
    high_quality = [r for r in results if r.self_consistency >= median_consistency]
    low_quality = [r for r in results if r.self_consistency < median_consistency]

    if high_quality and low_quality:
        print(f"\n--- High consistency group (n={len(high_quality)}) ---")
        print(f"  Mean collapses: {sum(r.collapse_count for r in high_quality)/len(high_quality):.1f}")
        print(f"  Mean divergence: {sum(r.mean_divergence for r in high_quality)/len(high_quality):.4f}")

        print(f"\n--- Low consistency group (n={len(low_quality)}) ---")
        print(f"  Mean collapses: {sum(r.collapse_count for r in low_quality)/len(low_quality):.1f}")
        print(f"  Mean divergence: {sum(r.mean_divergence for r in low_quality)/len(low_quality):.4f}")

        # Key correlation check
        hq_collapses = sum(r.collapse_count for r in high_quality) / len(high_quality)
        lq_collapses = sum(r.collapse_count for r in low_quality) / len(low_quality)
        if lq_collapses > hq_collapses:
            print(f"\n  >> Low-quality prompts have MORE collapses ({lq_collapses:.1f} vs {hq_collapses:.1f})")
            print("  >> This supports the thesis: structural defects precede quality loss")
        else:
            print(f"\n  >> No clear collapse-quality correlation ({lq_collapses:.1f} vs {hq_collapses:.1f})")

    # Save detailed results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for r in results:
        d = {
            "prompt": r.prompt,
            "self_consistency": r.self_consistency,
            "collapse_count": r.collapse_count,
            "drift_count": r.drift_count,
            "mean_divergence": r.mean_divergence,
            "first_instability_chunk": r.first_instability_chunk,
            "generation_time": r.generation_time,
            "samples": r.samples,
            "chunk_topologies": [asdict(c) for c in r.chunk_topologies],
        }
        serializable.append(d)
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
