import os
import pandas as pd
import glob

def analyze_cache(cache_dir="paper/results_cache/temp"):
    """
    Aggregates atomic benchmark results from the temporary cache and 
    prints a summary table of model performance.
    """
    csv_files = glob.glob(os.path.join(cache_dir, "fit_*.csv"))
    
    if not csv_files:
        print(f"No cached results found in {cache_dir}")
        return

    print(f"Aggregating {len(csv_files)} atomic results...")
    
    # Read and concatenate all individual result files
    df_list = []
    for f in csv_files:
        try:
            df_list.append(pd.read_csv(f))
        except Exception:
            continue
            
    if not df_list:
        print("Could not read any valid CSV files.")
        return
        
    full_df = pd.concat(df_list, ignore_index=True)
    
    # Key metrics to summarize
    metrics = ["Latent Recovery (U)", "Feature Recovery (V)", "Predictive Accuracy (Y)"]
    # Ensure columns exist before grouping
    available_metrics = [m for m in metrics if m in full_df.columns]
    
    # Group by Model, Consensus, and Loss to see the current state of the sweep
    summary = full_df.groupby(["Model", "Consensus", "Loss"])[available_metrics].agg(["mean", "std", "count"])
    
    # Sort for better readability (Best models at the top)
    if "Latent Recovery (U)" in available_metrics:
        summary = summary.sort_values(by=("Latent Recovery (U)", "mean"), ascending=False)

    print("\n" + "="*80)
    print("CURRENT CACHED BENCHMARK SUMMARY")
    print("="*80)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    print(summary)
    print("="*80)
    print(f"Total samples processed: {len(full_df)}")
    print("="*80)

if __name__ == "__main__":
    analyze_cache()
