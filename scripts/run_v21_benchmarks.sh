#!/bin/bash
# run_v21_benchmarks.sh
export PYTHONPATH="$(pwd)/src":$PYTHONPATH

# Explicitly set OMP to prevent hangs in background workers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Prevent matplotlib from trying to concurrently write to default cache dir
export MPLCONFIGDIR="$(pwd)/tmp_matplotlib_cache"
mkdir -p "$MPLCONFIGDIR"

# Disable OMP warnings about temp sizes
export KMP_WARNINGS=0

# CLEAR CACHE FOR FRESH RUN
rm -f paper/results_cache/unified_synthetic_v21.csv paper/results_cache/unified_real_v21.csv

echo "Starting Synthetic Benchmark v21 (5 seeds, ~1280 tasks) in background..."
python3 scripts/unified_benchmark.py --n-seeds 5 --workers 6 > synthetic_v21.log 2>&1 &
SYNTH_PID=$!

echo "Starting Real Benchmark v21 (5 seeds, ~640 tasks) in background..."
python3 scripts/unified_real_benchmark.py --n-seeds 5 --workers 6 > real_v21.log 2>&1 &
REAL_PID=$!

echo "Benchmarks PIDs: Synthetic=$SYNTH_PID, Real=$REAL_PID"
wait $SYNTH_PID $REAL_PID
echo "All v21 benchmarks complete."