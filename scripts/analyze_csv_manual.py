import os
import csv
import glob
from collections import defaultdict

def analyze_cache(cache_dir="paper/results_cache/temp"):
    csv_files = glob.glob(os.path.join(cache_dir, "fit_*.csv"))
    
    if not csv_files:
        print(f"No cached results found in {cache_dir}")
        return

    data = defaultdict(list)
    
    for f in csv_files:
        try:
            with open(f, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    key = (row['Model'], row['Consensus'], row['Loss'])
                    # Try to find common metrics
                    for metric in ["Latent Recovery (U)", "Feature Recovery (V)", "Predictive Accuracy (Y)"]:
                        if metric in row and row[metric]:
                            data[(key, metric)].append(float(row[metric]))
        except Exception as e:
            continue
            
    print(f"Aggregated {len(csv_files)} results.")
    print(f"{'Model':<10} | {'Consensus':<10} | {'Loss':<10} | {'Metric':<25} | {'Mean':<10} | {'Std':<10} | {'Count':<5}")
    print("-" * 100)
    
    for (key, metric) in sorted(data.keys()):
        vals = data[(key, metric)]
        if not vals: continue
        mean = sum(vals) / len(vals)
        std = (sum((x - mean)**2 for x in vals) / len(vals))**0.5
        count = len(vals)
        print(f"{key[0]:<10} | {key[1]:<10} | {key[2]:<10} | {metric:<25} | {mean:<10.4f} | {std:<10.4f} | {count:<5}")

if __name__ == "__main__":
    analyze_cache()
