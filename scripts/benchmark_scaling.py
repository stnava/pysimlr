import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pysimlr.deep import lend_simr

def benchmark_scaling():
    ns = [100, 500, 1000, 5000, 10000]
    ks = [2, 5, 10, 20]
    results = []

    for n in ns:
        for k in ks:
            # Simple linear data for benchmarking speed
            d = 100
            x1 = torch.randn(n, d)
            x2 = torch.randn(n, d)
            
            start_time = time.time()
            # Run for a few epochs to get a stable measure
            res = lend_simr([x1, x2], k=k, epochs=10, verbose=False)
            end_time = time.time()
            
            duration = (end_time - start_time) / 10.0 # average per epoch
            results.append({"n": n, "k": k, "TimePerEpoch": duration})
            print(f"n={n}, k={k}, time={duration:.4f}")

    df = pd.DataFrame(results)
    df.to_csv("paper/results_cache/scaling_benchmark.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="n", y="TimePerEpoch", hue="k", marker='o')
    plt.title("Execution Time Scaling (LEND)", fontweight='bold')
    plt.xlabel("Number of Samples (n)")
    plt.ylabel("Time per Epoch (seconds)")
    plt.grid(True)
    plt.savefig("paper/figures/scaling_plot.png")
    plt.close()

if __name__ == "__main__":
    benchmark_scaling()
