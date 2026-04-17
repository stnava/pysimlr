#!/bin/bash
# run_v21_benchmarks.sh
export PYTHONPATH="$(pwd)/src":$PYTHONPATH

echo "Starting Synthetic Benchmark v21..."
python3 scripts/unified_benchmark_v21.py --n-seeds 20 --workers 8

echo "Starting Real Benchmark v21..."
python3 scripts/unified_real_benchmark_v21.py --n-seeds 10 --workers 8

echo "All v21 benchmarks complete."
