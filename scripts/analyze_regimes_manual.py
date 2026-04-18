import os
import csv
from collections import defaultdict

def analyze_regime_files():
    files = {
        "Linear": "results_cache/benchmark_linear.csv",
        "Polynomial": "results_cache/benchmark_poly.csv",
        "Sine": "results_cache/benchmark_sin.csv",
        "Private": "results_cache/benchmark_private.csv"
    }
    
    for regime, path in files.items():
        if not os.path.exists(path):
            print(f"File {path} not found.")
            continue
            
        data = defaultdict(list)
        with open(path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                model = row.get('Model', row.get('Architecture', 'Unknown'))
                consensus = row.get('Consensus', 'Unknown')
                key = (model, consensus)
                
                for metric in ["Latent Recovery (U)", "Feature Recovery (V)"]:
                    if metric in row and row[metric]:
                        data[(key, metric)].append(float(row[metric]))
                        
        print(f"\nRegime: {regime}")
        print(f"{'Model':<20} | {'Consensus':<10} | {'Metric':<25} | {'Mean':<10} | {'Std':<10} | {'Count':<5}")
        print("-" * 100)
        
        for (key, metric) in sorted(data.keys()):
            vals = data[(key, metric)]
            mean = sum(vals) / len(vals)
            std = (sum((x - mean)**2 for x in vals) / len(vals))**0.5
            print(f"{key[0]:<20} | {key[1]:<10} | {metric:<25} | {mean:<10.4f} | {std:<10.4f} | {len(vals):<5}")

if __name__ == "__main__":
    analyze_regime_files()
