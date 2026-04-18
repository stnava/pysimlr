#!/bin/bash
# Run full comprehensive benchmark
export PYTHONPATH="$(pwd)/src":$PYTHONPATH

# Set OMP to prevent hangs
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Prevent matplotlib from trying to concurrently write to default cache dir
export MPLCONFIGDIR="$(pwd)/tmp_matplotlib_cache"
mkdir -p "$MPLCONFIGDIR"

# Disable OMP warnings about temp sizes
export KMP_WARNINGS=0

/Users/stnava/venvs/ants/bin/python -m benchmarks.runner --config configs/benchmark_full.yaml