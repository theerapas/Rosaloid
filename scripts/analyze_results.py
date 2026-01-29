import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Ensure src is in pythonpath
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def analyze(dms_id: str, results_dir: str):
    metrics_path = os.path.join(results_dir, f"{dms_id}_bo_metrics.csv")
    if not os.path.exists(metrics_path):
        print(f"No metrics found at {metrics_path}. Run BO first.")
        return

    df = pd.read_csv(metrics_path)
    print("Loaded metrics:")
    print(df)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Best so far
    axes[0].plot(df["round"], df["best_so_far"], marker='o')
    axes[0].set_title("Best Fitness Found")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Fitness (Max)")
    
    # 2. Diversity
    axes[1].plot(df["round"], df["diversity_mean_hamming"], marker='o', color='orange')
    axes[1].set_title("Batch Diversity (Hamming)")
    axes[1].set_xlabel("Round")
    
    # 3. Hit@96
    axes[2].plot(df["round"], df["Hit@96"], marker='o', color='green')
    axes[2].set_title("Hit@96 (Top True Found)")
    axes[2].set_xlabel("Round")
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"{dms_id}_plots.png")
    plt.savefig(plot_path)
    print(f"Plots saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dms_id", type=str, default="GFP_AEQVI_Sarkisyan_2016")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    
    analyze(args.dms_id, args.results_dir)

if __name__ == "__main__":
    main()
