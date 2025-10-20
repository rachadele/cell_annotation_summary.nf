#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style similar to other scripts
SMALL_SIZE = 15
MEDIUM_SIZE = 35
BIGGER_SIZE = 40
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot metrics for chosen pipeline parameters per study and celltype.")
    parser.add_argument('--weighted_metrics', type=str, help="Path to weighted metrics TSV file", default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/homo_sapiens/aggregated_results/weighted_f1_summary.tsv")
    parser.add_argument('--label_metrics', type=str, help="Path to label metrics TSV file", default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/homo_sapiens/aggregated_results/label_f1_summary.tsv")
    parser.add_argument('--subsample_ref', type=int, default=500, help="Subsample reference value")
    parser.add_argument('--cutoff', type=float, default=0.15, help="Cutoff value")
    parser.add_argument('--reference', type=str, default="whole_cortex", help="Reference name")
    parser.add_argument('--output', type=str, default="grant_figs.png", help="Output figure filename")
    # deal with jupyter kernel arguments
    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args

def filter_metrics(df, subsample_ref, cutoff, reference):
    # Filter for chosen parameters
    return df[(df['subsample_ref'] == subsample_ref) &
              (df['cutoff'] == cutoff) &
              (df['reference'] == reference)]


def plot_metrics(df, metric_col, title, ax):
    # Group by study and celltype, calculate mean and SD
    grouped = df.groupby(['study', 'celltype'])[metric_col].agg(['mean', 'std']).reset_index()
    # Plot mean with SD as error bars
    sns.barplot(data=grouped, x='study', y='mean', hue='celltype', ax=ax, ci=None)
    # Add error bars manually
    for i, row in grouped.iterrows():
        ax.errorbar(
            x=i,
            y=row['mean'],
            yerr=row['std'],
            fmt='none',
            c='black',
            capsize=5
        )
    ax.set_title(title)
    ax.set_ylabel(f"Mean {metric_col} ± SD")
    ax.set_xlabel('Study')
    ax.legend(title='Celltype', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()


def main():
    args = parse_arguments()
    weighted_df = pd.read_csv(args.weighted_metrics, sep='\t')
    label_df = pd.read_csv(args.label_metrics, sep='\t')

    weighted_filtered = filter_metrics(weighted_df, args.subsample_ref, args.cutoff, args.reference)
    label_filtered = filter_metrics(label_df, args.subsample_ref, args.cutoff, args.reference)

    # Get all unique levels
    levels = sorted(set(weighted_filtered['level']).union(set(label_filtered['level'])))
    n_levels = len(levels)
    fig, axes = plt.subplots(n_levels, 2, figsize=(18, 8 * n_levels))
    if n_levels == 1:
        axes = [axes]
    for idx, level in enumerate(levels):
        weighted_level = weighted_filtered[weighted_filtered['level'] == level]
        label_level = label_filtered[label_filtered['level'] == level]
        plot_metrics(weighted_level, 'weighted_f1', f'Weighted F1 per Study/Celltype ({level})', axes[idx][0])
        plot_metrics(label_level, 'label_f1', f'Label F1 per Study/Celltype ({level})', axes[idx][1])
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved figure to {args.output}")

if __name__ == "__main__":
    main()
