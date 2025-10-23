
#!/usr/bin/env python3

import argparse
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning)
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
    parser.add_argument('--remove_outliers', type=str, nargs='*', default=None,
                        help="List of study names to remove as outliers")
    parser.add_argument('--weighted_metrics', type=str, help="Path to weighted metrics TSV file", default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/mus_musculus/100/dataset_id/SCT/gap_false/aggregated_results/weighted_f1_results.tsv")
    parser.add_argument('--label_metrics', type=str, help="Path to label metrics TSV file", default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/mus_musculus/100/dataset_id/SCT/gap_false/aggregated_results/label_f1_results.tsv")
    parser.add_argument('--ref_keys', type=str, nargs='+', default=["subclass","class","family","global"], help="Reference keys to plot")
    parser.add_argument('--subsample_ref', type=int, default=500, help="Subsample reference value")
    parser.add_argument('--cutoff', type=float, default=0.15, help="Cutoff value")
    parser.add_argument('--reference', type=str, default="whole cortex", help="Reference name")
    parser.add_argument('--method', type=str, default="scvi", help="Method name")
    # deal with jupyter kernel arguments
    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args

def filter_metrics(df, subsample_ref, cutoff, reference, method):
    # Filter for chosen parameters
    return df[(df['subsample_ref'] == subsample_ref) &
              (df['cutoff'] == cutoff) &
              (df['reference'] == reference) &
              (df['method'] == method)]


def aggregate_metrics_long(df, groupby_col, metrics_to_agg):
    """
    Aggregates metrics by group, computes mean and std, and pivots to long format.
    """
    # Group by both groupby_col and 'key' (if present)
    group_cols = [groupby_col, 'key']
    stats = df.groupby(group_cols)[metrics_to_agg].agg(['mean', 'std']).reset_index()
    # Flatten columns
    stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in stats.columns.values]
    # Pivot longer
    df_long = stats.melt(id_vars=group_cols, var_name='metric_stat', value_name='value')
    df_long[['metric', 'stat']] = df_long['metric_stat'].str.rsplit('_', n=1, expand=True)
    df_long = df_long.pivot_table(index=group_cols + ['metric'], columns='stat', values='value').reset_index()
    return df_long

def plot_metrics(filtered_df, metric, ref_keys, group_col='study'):
    levels = ref_keys
    n_levels = len(levels)
    fig, axes = plt.subplots(n_levels, 1, figsize=(12, 6 * n_levels))
    if n_levels == 1:
        axes = [axes]
    mean_handle = None
    for idx, level in enumerate(levels):
        ax = axes[idx]
        level_df = filtered_df[filtered_df['key'] == level]
        # Add a label to the subplot indicating the level
        ax.set_title(f"Level: {level}", fontsize=16, loc='left')
        if not level_df.empty:
            # Use a palette with enough colors for all unique groups
            unique_groups = level_df[group_col].unique()
            n_colors = len(unique_groups)
            palette = sns.color_palette('tab20', n_colors=n_colors) if n_colors > 10 else sns.color_palette('tab10', n_colors=n_colors)
            sns.boxplot(data=level_df, y=group_col, x=metric, ax=ax, palette=palette, showfliers=False)
            means = level_df.groupby(group_col)[metric].mean().reset_index()
            for i, row in means.iterrows():
                sc = ax.scatter(row[metric], row[group_col], color='black', marker='o', s=60, zorder=3)
                if mean_handle is None:
                    mean_handle = sc
            # No y-axis label needed
            ax.set_ylabel('')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        # Set x-axis label for every subplot
        if idx == n_levels - 1:
            ax.set_xlabel(f'{metric} ± SD')
            ax.set_xticks(np.linspace(0, 1, num=6))
        else:
            ax.set_xlabel('')
            ax.set_xticks([])
        
    # Add a single legend for the mean to the last axis if any mean was plotted
    if mean_handle is not None:
        axes[-1].legend([mean_handle], ['Mean'], loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.suptitle(f'{metric} grouped by {group_col}', fontsize=25)
    plt.tight_layout()
    plt.savefig(f"{metric}_per_{group_col}.png")
    print(f"Saved figure to {metric}_per_{group_col}.png")


def main():
    args = parse_arguments()
    
    weighted_df = pd.read_csv(args.weighted_metrics, sep='\t')
    label_df = pd.read_csv(args.label_metrics, sep='\t')
    ref_keys = args.ref_keys

    # Remove outlier datasets if not None
    if args.remove_outliers:
        if 'study' in weighted_df.columns:
            weighted_df = weighted_df[~weighted_df['study'].isin(args.remove_outliers)]
        if 'label' in label_df.columns:
            label_df = label_df[~label_df['label'].isin(args.remove_outliers)]

    weighted_filtered = filter_metrics(weighted_df, args.subsample_ref, args.cutoff, args.reference, args.method)
    label_filtered = filter_metrics(label_df, args.subsample_ref, args.cutoff, args.reference, args.method)



    metrics_to_agg_weighted = ['weighted_f1', 'weighted_precision', 'weighted_recall', 'overall_accuracy']
    weighted_long = aggregate_metrics_long(
        weighted_filtered,
        groupby_col='study',
        metrics_to_agg=metrics_to_agg_weighted
    )
    weighted_long.to_csv("weighted_metrics_stats_per_study.tsv", sep="\t", index=False)

    metrics_to_agg_label = ['f1_score', 'precision', 'recall', 'accuracy']
    label_long = aggregate_metrics_long(
        label_filtered,
        groupby_col='label',
        metrics_to_agg=metrics_to_agg_label
        )
    label_long.to_csv("label_metrics_stats_per_label.tsv", sep="\t", index=False)


    plot_metrics(weighted_filtered, metric="weighted_f1", ref_keys=ref_keys, group_col='study')
    #plot_metrics(weighted_filtered, metric="macro_f1", ref_keys=ref_keys, group_col='study') # broken due to code error
    plot_metrics(weighted_filtered, metric="overall_accuracy", ref_keys=ref_keys, group_col='study')
    # precision and recall
    plot_metrics(weighted_filtered, metric="weighted_precision", ref_keys=ref_keys, group_col='study')
    plot_metrics(weighted_filtered, metric="weighted_recall", ref_keys=ref_keys, group_col='study')
    
    
    # drop missing rows
   # label_filtered = label_filtered.dropna(subset=["precision", "recall"])
    
    
    plot_metrics(label_filtered, metric="f1_score", ref_keys=ref_keys, group_col='label')
    # add accuracy, precision, recall plots here
    plot_metrics(label_filtered, metric="accuracy", ref_keys=ref_keys, group_col='label')


  
  #  plot_metrics(label_filtered, metric="precision", ref_keys=ref_keys, group_col='label')
  #  plot_metrics(label_filtered, metric="recall", ref_keys=ref_keys, group_col='label')

    # fails because precision is NA for some cell types due to lack of true positive predictions in that sample
    # this on a per-sample basis
    # computing overall precision and recall per cell type across samples addresses this
    # stored in another repo


if __name__ == "__main__":
    main()
