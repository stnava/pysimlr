#!/bin/bash
# run_v22_benchmarks.sh
#
# v22 re-runs the v21 protocol against the corrected library, with the seed
# count raised from 5 to 10.
#
# Why 10 seeds: inference is conducted at the level of the independent
# replicate (the seed), and the smallest attainable two-sided sign-flip
# permutation p-value is 2/2**n. At 5 seeds that floor is 0.0625, so no result
# could be reported below the conventional 0.05 threshold however strong the
# effect. At 10 seeds the floor is 2/1024 = 0.002. Crossing more loss functions
# or consensus algorithms does NOT add replication -- they are repeated
# measures on the same data splits.
set -euo pipefail

export PYTHONPATH="$(pwd)/src":${PYTHONPATH:-}

# Prevent thread oversubscription across the worker pool.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Give matplotlib a private cache so concurrent workers do not race on it.
export MPLCONFIGDIR="$(pwd)/tmp_matplotlib_cache"
mkdir -p "$MPLCONFIGDIR"

export KMP_WARNINGS=0

VERSION="${VERSION:-v22}"
SEEDS="${SEEDS:-10}"
WORKERS="${WORKERS:-6}"
PY="${PY:-python}"

echo "=== Benchmark ${VERSION}: ${SEEDS} seeds, ${WORKERS} workers ==="
"$PY" -c "
import sys; sys.path.insert(0, 'src')
from pysimlr.nsa_backend import backend_report
import pysimlr, torch
print(f'  pysimlr {pysimlr.__version__} | torch {torch.__version__}')
print(f'  nsa backend: {backend_report()}')
"

rm -f "paper/results_cache/unified_synthetic_${VERSION}.csv" \
      "paper/results_cache/unified_real_${VERSION}.csv"

echo "Starting synthetic benchmark in background..."
"$PY" scripts/unified_benchmark.py --version "$VERSION" --n-seeds "$SEEDS" \
    --workers "$WORKERS" > "synthetic_${VERSION}.log" 2>&1 &
SYNTH_PID=$!

echo "Starting real benchmark in background..."
"$PY" scripts/unified_real_benchmark.py --version "$VERSION" --n-seeds "$SEEDS" \
    --workers "$WORKERS" > "real_${VERSION}.log" 2>&1 &
REAL_PID=$!

echo "PIDs: synthetic=${SYNTH_PID}, real=${REAL_PID}"
wait $SYNTH_PID $REAL_PID
echo "=== ${VERSION} benchmarks complete ==="
wc -l "paper/results_cache/unified_synthetic_${VERSION}.csv" \
      "paper/results_cache/unified_real_${VERSION}.csv"
