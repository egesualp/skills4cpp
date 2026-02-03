import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

def visualize_grid_search(json_path, output_path):
    print(f"Loading data from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    results = []
    for entry in data['all_results']:
        config = entry['config']
        metrics = entry['metrics']
        
        row = {
            'weight': config['weight'],
            'temperature': config['temperature'],
            'threshold': config['threshold'],
            'map_full': metrics['map_full'],
            'mrr_full': metrics['mrr_full'],
            'recall@10': metrics.get('recall@10', 0),
            'recall@50': metrics.get('recall@50', 0)
        }
        results.append(row)

    df = pd.DataFrame(results)
    
    # We have 3 parameters: weight, temperature, threshold.
    # We'll plot heatmaps of weight vs temperature for each threshold.
    
    unique_thresholds = sorted(df['threshold'].unique())
    n_thresholds = len(unique_thresholds)
    
    # Setup the plot
    fig, axes = plt.subplots(1, n_thresholds, figsize=(5 * n_thresholds, 5), sharey=True)
    if n_thresholds == 1:
        axes = [axes]
        
    metric_to_plot = 'map_full'
    
    # Determine common vmin/vmax for consistent coloring
    vmin = df[metric_to_plot].min()
    vmax = df[metric_to_plot].max()

    print(f"Plotting {metric_to_plot} across {n_thresholds} thresholds...")

    for i, threshold in enumerate(unique_thresholds):
        subset = df[df['threshold'] == threshold]
        pivot_table = subset.pivot(index='temperature', columns='weight', values=metric_to_plot)
        
        sns.heatmap(pivot_table, ax=axes[i], annot=True, fmt=".3f", 
                    cmap="viridis", vmin=vmin, vmax=vmax, cbar=(i == n_thresholds - 1))
        
        axes[i].set_title(f'Threshold = {threshold}')
        axes[i].set_xlabel('Weight')
        if i == 0:
            axes[i].set_ylabel('Temperature')
        else:
            axes[i].set_ylabel('')

    plt.suptitle(f'Grid Search Results ({metric_to_plot})', y=1.05)
    plt.tight_layout()
    
    print(f"Saving plot to {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/bayesian_fusion/grid_search_metrics.json")
    parser.add_argument("--output_path", type=str, default="grid_search_visualization.png")
    args = parser.parse_args()
    
    visualize_grid_search(args.json_path, args.output_path)











