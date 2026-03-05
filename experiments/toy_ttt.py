"""Toy proof-of-concept: test-time LoRA refinement with topological loss.

This is a minimal experiment to validate the core loop works end-to-end:
1. Load a small model (default: Qwen3.5-0.8B)
2. Attach LoRA
3. Run TTT loop with PH loss on activations
4. Verify that LoRA weights change and loss decreases

Run:
    python experiments/toy_ttt.py
    python experiments/toy_ttt.py --model Qwen/Qwen3.5-2B --steps 3
    python experiments/toy_ttt.py --model gpt2 --steps 5  # fallback for CPU-only
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from livelora.core.lora_adapter import LiveLoraConfig, LiveLoraModel, get_lora_target_modules
from livelora.core.ttt_loop import TTTConfig, TTTLoop
from livelora.topology.ph_loss import DifferentiablePHLoss

# Default model: Qwen3.5-0.8B — small enough for CPU, modern architecture
DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"


def main():
    parser = argparse.ArgumentParser(description="Toy TTT experiment")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model name")
    parser.add_argument("--steps", type=int, default=5, help="Number of TTT steps")
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--prompt", default="The topology of neural representations", help="Input")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--dtype", default="auto", help="Model dtype (auto/float32/bfloat16)")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Resolve dtype — use bfloat16 on GPU for Qwen3.5, float32 on CPU
    if args.dtype == "auto":
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    else:
        dtype = getattr(torch, args.dtype)

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Model: {args.model}")
    print(f"LoRA rank: {args.rank}, LR: {args.lr}, Steps: {args.steps}")
    print()

    # Load model + tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
    )
    if device == "cpu":
        base_model = base_model.to(device)

    # Wrap with LiveLoRA — auto-detect target modules for the model architecture
    target_modules = get_lora_target_modules(base_model)
    lora_config = LiveLoraConfig(
        rank=args.rank,
        target_modules=target_modules,
    )
    print(f"LoRA target modules: {target_modules}")
    model = LiveLoraModel(base_model, lora_config)

    # Count params
    lora_params = sum(p.numel() for p in model.lora_parameters())
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"LoRA params: {lora_params:,} / {total_params:,} total ({100*lora_params/total_params:.2f}%)")

    # Set up PH loss and TTT loop
    ph_loss = DifferentiablePHLoss(
        max_dimension=1,
        max_points=64,
        target_betti={0: 1, 1: 0},  # Want one connected component, no loops
    )

    ttt_config = TTTConfig(
        num_steps=args.steps,
        lr=args.lr,
        target_layers=[-1, -2],  # Last two layers
        activation_subsample=32,
        adapt_and_reset=False,  # Keep changes to see them accumulate
    )

    ttt = TTTLoop(model, ph_loss, ttt_config)

    # Tokenize input
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    print(f"\nInput: '{args.prompt}' ({inputs['input_ids'].shape[1]} tokens)")

    # Snapshot initial LoRA weights
    initial_lora_norm = sum(
        p.data.norm().item() for p in model.lora_parameters()
    )

    # Run TTT
    print(f"\nRunning {args.steps} TTT steps...")
    t0 = time.perf_counter()
    losses = ttt.refine(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
    )
    elapsed = time.perf_counter() - t0

    # Report
    final_lora_norm = sum(
        p.data.norm().item() for p in model.lora_parameters()
    )

    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f}s ({elapsed/args.steps:.2f}s/step)")
    print(f"  Losses: {[f'{l:.4f}' for l in losses]}")
    print(f"  LoRA weight norm: {initial_lora_norm:.4f} -> {final_lora_norm:.4f}")
    print(f"  Weight change: {abs(final_lora_norm - initial_lora_norm):.6f}")

    if losses[-1] < losses[0]:
        print("  Loss decreased — TTT loop is working!")
    else:
        print("  Loss did not decrease — may need LR/config tuning.")

    # Generate before/after
    print("\n--- Generation (after TTT) ---")
    model.eval()
    with torch.no_grad():
        gen_ids = model.model.generate(
            **inputs, max_new_tokens=50, do_sample=False,
        )
    print(tokenizer.decode(gen_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
