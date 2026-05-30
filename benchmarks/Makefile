# TorchFX benchmark orchestration.
#
# Self-contained: this Makefile only depends on what's in the main repo
# (benchmarks/, tools/, src/). It does NOT depend on any downstream
# consumer (e.g. an external paper repository) and is safe to ship to
# the SLURM cluster on its own.
#
# What's in here
# --------------
# make bench-cpu          single-filter, pipeline, FFT-conv, hotpath, design
# make bench-cuda         same set, restricted to GPU benchmarks (needs CUDA)
# make bench-comparators  torchaudio head-to-head (skips if torchaudio missing)
# make bench-realtime     deterministic realtime callback latency / jitter
# make launches           fused-vs-unfused dispatch + wall-time (CPU)
# make launches-cuda      same, on CUDA
# make threshold-cuda     PARALLEL_SCAN_THRESHOLD ablation
# make aggregate          pp50/p95/p99/IQR table + CSV across all JSON in $(RESULTS)
# make all                bench-cpu + bench-realtime + bench-comparators + launches + aggregate
# make all-cuda           the CUDA-side counterpart
# make clean              wipe $(RESULTS)
#
# Output layout
# -------------
# Every target writes JSON into $(RESULTS) (default ``benchmarks/results/``)
# tagged with ``hostname -s`` so multi-machine runs coexist without
# overwriting each other:
#
#     cpu-<host>.json
#     cuda-<host>.json
#     comparators-<host>.json
#     realtime-<host>.json
#     launches-cpu-<host>.json
#     launches-cuda-<host>.json
#     threshold-cuda-<host>.json
#     summary.csv
#     summary.txt
#
# Overrides
# ---------
# RESULTS=/path/to/dir        change output directory
# HOST=customname             override the host tag
# BENCH_FLAGS="..."           extra pytest-benchmark flags
# ENERGY=1                    instrument bench targets with tools/energy_meter.py
#                             (writes energy-<target>-<host>.json alongside)
#
# Example: cluster run with results in /ext/scratch
#     make -C benchmarks all-cuda RESULTS=/ext/scratch/torchfx-results

# Host tag for output filenames. Combines hostname with GPU model when
# nvidia-smi is available so multi-machine / multi-GPU runs are
# distinguishable in benchmarks/results/. Examples:
#   gpu7-L40S, gpu5-A40, gpu2-RTX-3090, Alienware-RTX-3070, my-laptop
# Falls back to plain hostname on CPU-only nodes.
#
# Some HPC nodes return platform tags (e.g. x86_64-conda-linux-gnu)
# instead of real hostnames from `hostname -s` when run inside a
# conda activation script; appending the GPU model keeps the tag
# meaningful even when the hostname is uninformative.
HOST           ?= $(shell \
                    h=$$(hostname -s 2>/dev/null || echo host); \
                    g=$$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null \
                          | head -1 \
                          | sed -E 's/^NVIDIA //; s/^GeForce //; s/[[:space:]]+/-/g'); \
                    if [ -n "$$g" ]; then echo "$${h}-$${g}"; else echo "$$h"; fi \
                  )
ROOT           := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PROJECT        := $(abspath $(ROOT)/..)
RESULTS        ?= $(ROOT)/results
PYTHON         ?= uv run --no-sync python
PYTEST         ?= uv run --no-sync pytest
TOOLS_DIR      := $(PROJECT)/tools
BENCH_DIR      := $(ROOT)

# pytest-benchmark flags. ``--benchmark-enable`` is required because the
# project sets ``--benchmark-disable`` in pyproject.toml's addopts.
BENCH_FLAGS    ?= --benchmark-enable --benchmark-min-rounds=20 --benchmark-warmup=on -q

# Energy meter wrapper. ``ENERGY=1`` runs each bench target inside
# tools/energy_meter.py so cpu_joules / gpu_joules / duration are
# recorded alongside the bench JSON.
define _energy_wrap
$(if $(ENERGY),$(PYTHON) $(TOOLS_DIR)/energy_meter.py --out $(RESULTS)/energy-$(1)-$(HOST).json --,)
endef

.PHONY: all all-cuda clean dirs bench-cpu bench-cuda bench-comparators bench-realtime \
        launches launches-cuda threshold-cuda aggregate help

help:
	@awk '/^# make / { sub("^# *", ""); print }' $(lastword $(MAKEFILE_LIST))

all: dirs bench-cpu bench-realtime bench-comparators launches aggregate

# Convenience: everything that requires CUDA. Use on a GPU node.
all-cuda: dirs bench-cuda bench-comparators launches-cuda threshold-cuda aggregate

dirs:
	@mkdir -p $(RESULTS)

# -----------------------------------------------------------------------------
# Core CPU benchmarks: single-filter, pipeline, FFT-conv, hotpath.
# -----------------------------------------------------------------------------
bench-cpu: dirs
	@echo "[bench-cpu] $(HOST) -> $(RESULTS)/cpu-$(HOST).json"
	$(call _energy_wrap,bench-cpu) \
	$(PYTEST) $(BENCH_DIR)/test_iir_bench.py \
	          $(BENCH_DIR)/test_biquad_bench.py \
	          $(BENCH_DIR)/test_fir_bench.py \
	          $(BENCH_DIR)/test_fftconv_bench.py \
	          $(BENCH_DIR)/test_pipeline_bench.py \
	          $(BENCH_DIR)/test_hotpath_bench.py \
	          $(BENCH_DIR)/test_api_bench.py \
	          $(BENCH_DIR)/test_design_benchmarks.py \
	          -k "not numba_cuda and not gpu" \
	          $(BENCH_FLAGS) \
	          --benchmark-json=$(RESULTS)/cpu-$(HOST).json

# -----------------------------------------------------------------------------
# CUDA benchmarks. Skipped if torch.cuda.is_available() is False — the
# individual tests already use pytest.skip in that case.
# -----------------------------------------------------------------------------
bench-cuda: dirs
	@echo "[bench-cuda] $(HOST) -> $(RESULTS)/cuda-$(HOST).json"
	$(call _energy_wrap,bench-cuda) \
	$(PYTEST) $(BENCH_DIR)/test_iir_bench.py \
	          $(BENCH_DIR)/test_biquad_bench.py \
	          $(BENCH_DIR)/test_fir_bench.py \
	          $(BENCH_DIR)/test_fftconv_bench.py \
	          $(BENCH_DIR)/test_pipeline_bench.py \
	          $(BENCH_DIR)/test_hotpath_bench.py \
	          -k "gpu or cuda" \
	          $(BENCH_FLAGS) \
	          --benchmark-json=$(RESULTS)/cuda-$(HOST).json

# -----------------------------------------------------------------------------
# Comparator benchmarks: torchaudio head-to-head. Skips cleanly if
# torchaudio is not installed.
# -----------------------------------------------------------------------------
bench-comparators: dirs
	@echo "[bench-comparators] $(HOST) -> $(RESULTS)/comparators-$(HOST).json"
	$(call _energy_wrap,bench-comparators) \
	$(PYTEST) $(BENCH_DIR)/test_torchaudio_bench.py \
	          $(BENCH_FLAGS) \
	          --benchmark-json=$(RESULTS)/comparators-$(HOST).json

# -----------------------------------------------------------------------------
# Realtime benchmarks: per-callback latency + jitter distribution.
# -----------------------------------------------------------------------------
bench-realtime: dirs
	@echo "[bench-realtime] $(HOST) -> $(RESULTS)/realtime-$(HOST).json"
	$(call _energy_wrap,bench-realtime) \
	$(PYTEST) $(BENCH_DIR)/test_realtime_bench.py \
	          $(BENCH_FLAGS) \
	          --benchmark-json=$(RESULTS)/realtime-$(HOST).json

# -----------------------------------------------------------------------------
# Kernel-launch / dispatch counts: fused vs unfused IIR cascade.
# -----------------------------------------------------------------------------
launches: dirs
	@echo "[launches] $(HOST) -> $(RESULTS)/launches-cpu-$(HOST).json"
	$(PYTHON) $(TOOLS_DIR)/count_kernel_launches.py \
	          --depths 2 5 10 20 50 \
	          --duration 5.0 \
	          --channels 2 \
	          --fs 48000 \
	          --device cpu \
	          --iters 30 \
	          --out $(RESULTS)/launches-cpu-$(HOST).json

launches-cuda: dirs
	@echo "[launches-cuda] $(HOST) -> $(RESULTS)/launches-cuda-$(HOST).json"
	$(PYTHON) $(TOOLS_DIR)/count_kernel_launches.py \
	          --depths 2 5 10 20 50 \
	          --duration 5.0 \
	          --channels 2 \
	          --fs 48000 \
	          --device cuda \
	          --iters 30 \
	          --out $(RESULTS)/launches-cuda-$(HOST).json

# PARALLEL_SCAN_THRESHOLD ablation. CUDA-only.
threshold-cuda: dirs
	@echo "[threshold-cuda] $(HOST) -> $(RESULTS)/threshold-cuda-$(HOST).json"
	$(PYTHON) $(TOOLS_DIR)/threshold_sweep.py \
	          --device cuda \
	          --channels 1 2 8 32 \
	          --iters 50 \
	          --out $(RESULTS)/threshold-cuda-$(HOST).json

# -----------------------------------------------------------------------------
# Aggregate: pull every pytest-benchmark JSON in $(RESULTS) into one
# table (p50/p95/p99/IQR per benchmark) + CSV. Non-pytest-benchmark JSON
# (launches, threshold, energy) is skipped with a warning.
# -----------------------------------------------------------------------------
aggregate: dirs
	@echo "[aggregate] $(HOST) -> $(RESULTS)/summary.{csv,txt}"
	@JSON_FILES=$$(ls $(RESULTS)/*.json 2>/dev/null); \
	if [ -n "$$JSON_FILES" ]; then \
		$(PYTHON) $(TOOLS_DIR)/aggregate_benchmarks.py \
		          $$JSON_FILES \
		          --format csv \
		          --out $(RESULTS)/summary.csv; \
		$(PYTHON) $(TOOLS_DIR)/aggregate_benchmarks.py \
		          $$JSON_FILES \
		          --format table > $(RESULTS)/summary.txt; \
		echo "[ok] $(RESULTS)/summary.csv"; \
		echo "[ok] $(RESULTS)/summary.txt"; \
	else \
		echo "[skip] no JSON files in $(RESULTS)"; \
	fi

clean:
	rm -rf $(RESULTS)
