#!/bin/bash
#SBATCH --job-name=torchfx-bench
#SBATCH --output=benchmarks/results/slurm-%j.out
#SBATCH --error=benchmarks/results/slurm-%j.err
#SBATCH --gres=gpu:rtx
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

# ── Setup ────────────────────────────────────────────────────────────────────
RESULTS_DIR="benchmarks/results/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

echo "=== TorchFX Benchmark Run ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Date:      $(date -Iseconds)"
echo "Results:   $RESULTS_DIR"
echo ""

# Log environment info
{
    echo "hostname: $(hostname)"
    echo "date: $(date -Iseconds)"
    echo "slurm_job_id: $SLURM_JOB_ID"
    python -c "
import torch
print(f'torch: {torch.__version__}')
print(f'cuda_available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'gpu: {torch.cuda.get_device_name(0)}')
    print(f'gpu_memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
} > "$RESULTS_DIR/environment.txt"

cat "$RESULTS_DIR/environment.txt"
echo ""

# ── Run benchmarks ───────────────────────────────────────────────────────────

# IIR filter benchmarks (GPU + CPU + SciPy)
echo ">>> Running IIR benchmarks..."
uv run pytest benchmarks/test_iir_bench.py \
    --benchmark-enable \
    --benchmark-json="$RESULTS_DIR/iir_bench.json" \
    --benchmark-name=short \
    -v --tb=short 2>&1 | tee "$RESULTS_DIR/iir_bench.log"

# FIR filter benchmarks
echo ">>> Running FIR benchmarks..."
uv run pytest benchmarks/test_fir_bench.py \
    --benchmark-enable \
    --benchmark-json="$RESULTS_DIR/fir_bench.json" \
    --benchmark-name=short \
    -v --tb=short 2>&1 | tee "$RESULTS_DIR/fir_bench.log"

# API comparison benchmarks
echo ">>> Running API benchmarks..."
uv run pytest benchmarks/test_api_bench.py \
    --benchmark-enable \
    --benchmark-json="$RESULTS_DIR/api_bench.json" \
    --benchmark-name=short \
    -v --tb=short 2>&1 | tee "$RESULTS_DIR/api_bench.log"

# CUDA-specific benchmarks (stateful, SOS cascade, pipeline)
echo ">>> Running CUDA benchmarks..."
uv run pytest benchmarks/test_cuda_bench.py \
    --benchmark-enable \
    --benchmark-json="$RESULTS_DIR/cuda_bench.json" \
    --benchmark-name=short \
    -v --tb=short 2>&1 | tee "$RESULTS_DIR/cuda_bench.log"

# ── Combine results ──────────────────────────────────────────────────────────
echo ""
echo "=== Benchmark run complete ==="
echo "Results saved to: $RESULTS_DIR"
ls -lh "$RESULTS_DIR"
