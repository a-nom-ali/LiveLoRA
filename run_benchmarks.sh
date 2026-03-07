#!/bin/bash
# LiveLoRA Benchmark Runner
#
# Run correctness benchmarks on a GPU machine and commit results.
# Designed to be run by someone without Claude access.
#
# Usage:
#   ./run_benchmarks.sh                    # auto-detect GPU, run all
#   ./run_benchmarks.sh --model Qwen/Qwen3.5-9B --quantize 4bit
#   ./run_benchmarks.sh --model Qwen/Qwen3.5-27B --quantize 4bit
#   ./run_benchmarks.sh --skip-install     # skip pip install step
#
# Results are saved to outputs/results/ which is tracked by git.
# After running, just: git add outputs/results/ && git commit && git push

set -e

# Defaults
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
QUANTIZE="${QUANTIZE:-none}"
NUM_PROBLEMS="${NUM_PROBLEMS:-50}"
N_SAMPLES="${N_SAMPLES:-3}"
DEVICE="${DEVICE:-auto}"
SKIP_INSTALL=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2;;
        --quantize) QUANTIZE="$2"; shift 2;;
        --num-problems) NUM_PROBLEMS="$2"; shift 2;;
        --n-samples) N_SAMPLES="$2"; shift 2;;
        --device) DEVICE="$2"; shift 2;;
        --skip-install) SKIP_INSTALL=true; shift;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

# Derive a safe name for output files
SAFE_MODEL=$(echo "$MODEL" | sed 's/[\/:]/_/g')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
QUANT_SUFFIX=""
if [ "$QUANTIZE" != "none" ]; then
    QUANT_SUFFIX="_${QUANTIZE}"
fi

echo "=============================================="
echo "LiveLoRA Benchmark Runner"
echo "=============================================="
echo "Model:        $MODEL"
echo "Quantize:     $QUANTIZE"
echo "Problems:     $NUM_PROBLEMS"
echo "Samples:      $N_SAMPLES"
echo "Device:       $DEVICE"
echo "Timestamp:    $TIMESTAMP"
echo "=============================================="

# Check GPU
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f}GB' if torch.cuda.is_available() else '')" 2>/dev/null || echo "Warning: could not detect GPU"

# Note: Models are downloaded automatically from HuggingFace on first run.
# Qwen3.5 models are open-access (no token needed).
# Download sizes: 9B ~18GB, 27B ~54GB (4-bit loads are smaller in VRAM but full download).
# To change download location: export HF_HOME=/path/to/big/drive/huggingface_cache
echo "HF cache: ${HF_HOME:-~/.cache/huggingface}"

# Install if needed
if [ "$SKIP_INSTALL" = false ]; then
    echo ""
    echo "Installing dependencies..."
    pip install -e ".[dev]" --quiet
    pip install bitsandbytes datasets --quiet
    echo "Done."
fi

# Build quantize flag
QUANT_FLAG=""
if [ "$QUANTIZE" != "none" ]; then
    QUANT_FLAG="--quantize $QUANTIZE"
fi

# Run ARC-Challenge
echo ""
echo "=============================================="
echo "Running ARC-Challenge benchmark..."
echo "=============================================="
ARC_OUTPUT="outputs/results/arc_challenge_${SAFE_MODEL}${QUANT_SUFFIX}_${TIMESTAMP}.json"
python3 experiments/arc_benchmark.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --num-problems "$NUM_PROBLEMS" \
    --n-samples "$N_SAMPLES" \
    $QUANT_FLAG \
    --output "$ARC_OUTPUT"

echo ""
echo "ARC results saved to: $ARC_OUTPUT"

# Run GSM8K
echo ""
echo "=============================================="
echo "Running GSM8K benchmark..."
echo "=============================================="
GSM_OUTPUT="outputs/results/gsm8k_${SAFE_MODEL}${QUANT_SUFFIX}_${TIMESTAMP}.json"
python3 experiments/gsm8k_benchmark.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --num-problems "$NUM_PROBLEMS" \
    --n-samples "$N_SAMPLES" \
    $QUANT_FLAG \
    --output "$GSM_OUTPUT"

echo ""
echo "GSM8K results saved to: $GSM_OUTPUT"

# Summary
echo ""
echo "=============================================="
echo "ALL BENCHMARKS COMPLETE"
echo "=============================================="
echo ""
echo "Results:"
echo "  $ARC_OUTPUT"
echo "  $GSM_OUTPUT"
echo ""
echo "To share results, run:"
echo "  git add outputs/results/"
echo "  git commit -m 'benchmark results: $MODEL $QUANTIZE'"
echo "  git push"
