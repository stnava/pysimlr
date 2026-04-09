#!/bin/bash
# Run small benchmark for quick verification
export PYTHONPATH="$(pwd)/src":$PYTHONPATH
/Users/stnava/venvs/ants/bin/python -m benchmarks.runner --config configs/benchmark_smoke.yaml
