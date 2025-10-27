

#!/usr/bin/env python3

import argparse
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
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
    parser.add_argument('--remove_outliers', type=str, nargs='*', default=["GSE180670"],
                        help="List of study names to remove as outliers")
    parser.add_argument('--weighted_metrics', type=str, help="Path to weighted metrics TSV file", default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/homo_sapiens/100/dataset_id/SCT/gap_false/aggregated_results/weighted_f1_results.tsv")
    parser.add_argument('--label_metrics', type=str, help="Path to label metrics TSV file", default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/homo_sapiens/100/dataset_id/SCT/gap_false/aggregated_results/label_f1_results.tsv")
    parser.add_argument('--ref_keys', type=str, nargs='+', default=["subclass","class","family"], help="Reference keys to plot")
    parser.add_argument('--subsample_ref', type=int, default=500, help="Subsample reference value")
    parser.add_argument('--cutoff', type=float, default=0, help="Cutoff value")
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


def plot_metrics(filtered_df, metric, ref_keys, group_col='study', outdir="outliers_kept"):
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
           # unique_groups = level_df[group_col].unique()
           # n_colors = len(unique_groups)
           # palette = sns.color_palette('tab20', n_colors=n_colors) if n_colors > 10 else sns.color_palette('tab10', n_colors=n_colors)
            sns.boxplot(data=level_df, y=group_col, x=metric, ax=ax, showfliers=False)
            means = level_df.groupby(group_col)[metric].mean().reset_index()
            for i, row in means.iterrows():
                sc = ax.scatter(row[metric], row[group_col], color='black', marker='o', s=60, zorder=3)
                if mean_handle is None:
                    mean_handle = sc
            # No y-axis label needed
            ax.set_ylabel('')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        # Set x-axis label and enforce range/ticks for every subplot
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks(np.arange(-0.05, 1.05, 0.2))
        if idx == n_levels - 1:
            ax.set_xlabel(f'{metric} ± SD')
        else:
            ax.set_xlabel('')
        
    # Add a single legend for the mean to the last axis if any mean was plotted
    if mean_handle is not None:
        axes[-1].legend([mean_handle], ['Mean'], loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.suptitle(f'{metric} grouped by {group_col}', fontsize=25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,f"{metric}_per_{group_col}.png"))
    plt.close()



def dotplot(df, x, y, color_var, title=None, xlabel=None, ylabel=None, outdir=None):
    plt.figure(figsize=(10, 10))
    # Plot only dots for each cell type (y axis), colored by study, small size
    sns.stripplot(data=df, x=y, y=x, hue=color_var, dodge=False, palette='tab10', size=3, alpha=0.8)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(ylabel if ylabel else y)
    if ylabel:
        plt.ylabel(xlabel if xlabel else x)
    plt.ylim(-0.05, 1.05)
    plt.yticks(np.arange(-0.05, 1.06, 0.2))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{x}_by_{y}_dots.png"))
    plt.close()


def main():


    args = parse_arguments()
    weighted_df = pd.read_csv(args.weighted_metrics, sep='\t')
    label_df = pd.read_csv(args.label_metrics, sep='\t')
    ref_keys = args.ref_keys
    # Determine output directory based on outlier removal
    if args.remove_outliers and len(args.remove_outliers) > 0:
        outdir = 'outliers_removed'
    else:
        outdir = 'outliers_kept'
    os.makedirs(outdir, exist_ok=True)
    # Remove outlier datasets if not None
    if args.remove_outliers:
        if 'study' in weighted_df.columns:
            weighted_df = weighted_df[~weighted_df['study'].isin(args.remove_outliers)]
        if 'label' in label_df.columns:
            label_df = label_df[~label_df['label'].isin(args.remove_outliers)]

    weighted_filtered = filter_metrics(weighted_df, args.subsample_ref, args.cutoff, args.reference, args.method)
    label_filtered = filter_metrics(label_df, args.subsample_ref, args.cutoff, args.reference, args.method)

    # drop "Glia" and "VLMCeuron" - hack for now
    label_filtered = label_filtered[~label_filtered['label'].isin(["Glia", "VLMCeuron"])]

    metrics_to_agg_weighted = ['weighted_f1', 
                               'weighted_precision', 
                               'weighted_recall', 
                               'overall_accuracy', 
                               'macro_f1', 
                               'macro_precision', 
                               'macro_recall',
                               'micro_f1',
                               'micro_precision',
                               'micro_recall']
    weighted_long = aggregate_metrics_long(
        weighted_filtered,
        groupby_col='study',
        metrics_to_agg=metrics_to_agg_weighted
    )
    weighted_long.to_csv(f"{outdir}/weighted_metrics_stats_per_study.tsv", sep="\t", index=False)

    metrics_to_agg_label = ['f1_score', 'precision', 'recall']
    label_long = aggregate_metrics_long(
        label_filtered,
        groupby_col='label',
        metrics_to_agg=metrics_to_agg_label
        )
    label_long.to_csv(f"{outdir}/label_metrics_stats_per_label.tsv", sep="\t", index=False)


    # change this to a loop over metrics
    # plot weighted metrics
    for metric in metrics_to_agg_weighted:
        if metric in weighted_filtered.columns:
            plot_metrics(weighted_filtered, metric=metric, ref_keys=ref_keys, group_col='study', outdir=outdir)
            plt.close()



    plot_metrics(label_filtered, metric="f1_score", ref_keys=ref_keys, group_col='label', outdir=outdir)
    plt.close()
    # add accuracy, precision, recall plots here
    # Plot dots for label-level metrics, colored by study
    dotplot(
        label_filtered,
        x="f1_score",
        y='label',
        color_var='study',
        title=f'f1_score by label (colored by study)',
        xlabel='f1_score',
        ylabel='Label',
        outdir=outdir
    )
  
    # plot_metrics(label_filtered, metric="accuracy", ref_keys=ref_keys, group_col='label')
  # plot_metrics(label_filtered, metric="precision", ref_keys=ref_keys, group_col='label')
  #  plot_metrics(label_filtered, metric="recall", ref_keys=ref_keys, group_col='label')

    # fails because precision is NA for some cell types due to lack of true positive predictions in that sample
    # this on a per-sample basis
    # computing overall precision and recall per cell type across samples addresses this
    # stored in another repo

    # Summary of means and SDs for all metrics across all studies (weighted)
    weighted_summary = weighted_filtered[metrics_to_agg_weighted].agg(['mean', 'std']).T.reset_index()
    weighted_summary.columns = ['metric', 'mean', 'std']
    weighted_summary.to_csv(f'{outdir}/weighted_metrics_summary_overall.tsv', sep='\t', index=False)

    # Summary of means and SDs for all metrics across all cell types (label)
    label_summary = label_filtered[metrics_to_agg_label].agg(['mean', 'std']).T.reset_index()
    label_summary.columns = ['metric', 'mean', 'std']
    label_summary.to_csv(f'{outdir}/label_metrics_summary_overall.tsv', sep='\t', index=False)

if __name__ == "__main__":
    main()
