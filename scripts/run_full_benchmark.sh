#!/bin/bash
# Run full comprehensive benchmark
export PYTHONPATH="$(pwd)/src":$PYTHONPATH
/Users/stnava/venvs/ants/bin/python -m benchmarks.runner --config configs/benchmark_full.yaml
