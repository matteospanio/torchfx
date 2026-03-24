#!/bin/bash
#SBATCH --job-name=tfx-bench
#SBATCH --output=benchmarks/results/slurm-%j.out
#SBATCH --error=benchmarks/results/slurm-%j.err
#SBATCH --gres=gpu:rtx
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=08:00:00

set -euo pipefail

module load GCC
module load CUDA

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
    uv run python -c "
import torch
print(f'torch: {torch.__version__}')
print(f'cuda_available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'gpu: {torch.cuda.get_device_name(0)}')
    print(f'gpu_memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
try:
    from numba import cuda as nc
    print(f'numba_cuda: {nc.is_available()}')
except ImportError:
    print('numba_cuda: not installed')
"
} > "$RESULTS_DIR/environment.txt"

cat "$RESULTS_DIR/environment.txt"
echo ""

BENCH_OPTS="--benchmark-enable --benchmark-name=short -v --tb=short"

# ── Run benchmarks ───────────────────────────────────────────────────────────

echo ">>> Running IIR benchmarks..."
uv run pytest benchmarks/test_iir_bench.py \
    $BENCH_OPTS --benchmark-json="$RESULTS_DIR/iir_bench.json" \
    2>&1 | tee "$RESULTS_DIR/iir_bench.log"

echo ">>> Running FIR benchmarks..."
uv run pytest benchmarks/test_fir_bench.py \
    $BENCH_OPTS --benchmark-json="$RESULTS_DIR/fir_bench.json" \
    2>&1 | tee "$RESULTS_DIR/fir_bench.log"

echo ">>> Running FFT conv benchmarks..."
uv run pytest benchmarks/test_fftconv_bench.py \
    $BENCH_OPTS --benchmark-json="$RESULTS_DIR/fftconv_bench.json" \
    2>&1 | tee "$RESULTS_DIR/fftconv_bench.log"

echo ">>> Running biquad benchmarks..."
uv run pytest benchmarks/test_biquad_bench.py \
    $BENCH_OPTS --benchmark-json="$RESULTS_DIR/biquad_bench.json" \
    2>&1 | tee "$RESULTS_DIR/biquad_bench.log"

echo ">>> Running pipeline benchmarks..."
uv run pytest benchmarks/test_pipeline_bench.py \
    $BENCH_OPTS --benchmark-json="$RESULTS_DIR/pipeline_bench.json" \
    2>&1 | tee "$RESULTS_DIR/pipeline_bench.log"

echo ">>> Running API benchmarks..."
uv run pytest benchmarks/test_api_bench.py \
    $BENCH_OPTS --benchmark-json="$RESULTS_DIR/api_bench.json" \
    2>&1 | tee "$RESULTS_DIR/api_bench.log"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Benchmark run complete ==="
echo "Results saved to: $RESULTS_DIR"
ls -lh "$RESULTS_DIR"
