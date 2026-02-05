

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
MEDIUM_SIZE = 20
BIGGER_SIZE = 40
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

human_pmid_mapping = {
    "CMC": "38781388",
    "DevBrain": "38781369.1",
    "GSE211870": "37910626",
    "GSE237718": "40112813",
    "Lau": "32989152",
    "Lim": "36543778",
    "Ling-2024": "38448582",
    "MultiomeBrain": "38781369.2",
    "Nagy": "32341540",
    "Pineda": "38521060",
    "PTSDBrainomics": "38781393",
    "Rosmap": "31042697",
    "UCLA-ASD": "31220268",
    "Velmeshev-et-al.1": "31097668.1",
    "Velmeshev-et-al.2": "31097668.2"
}

def parse_arguments():

    parser = argparse.ArgumentParser(description="Plot metrics for chosen pipeline parameters per study and celltype.")
    parser.add_argument('--remove_outliers', type=str, nargs='*', default=["GSE180670"], help="List of study names to remove as outliers")
    parser.add_argument('--weighted_metrics', type=str, help="Path to weighted metrics TSV file", default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/homo_sapiens/100/dataset_id/SCT/gap_false/aggregated_results/sample_results.tsv")
    parser.add_argument('--label_metrics', type=str, help="Path to label metrics TSV file", default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/homo_sapiens/100/dataset_id/SCT/gap_false/aggregated_results/label_results.tsv")
    parser.add_argument('--ref_keys', type=str, nargs='+', default=["subclass","class","family"], help="Reference keys to plot")
    parser.add_argument('--subsample_ref', type=int, default=500, help="Subsample reference value")
    parser.add_argument('--cutoff', type=float, default=0, help="Cutoff value")
    parser.add_argument('--reference', type=str, default="whole cortex", help="Reference name")
    parser.add_argument('--method', type=str, default="scvi", help="Method name")
    parser.add_argument('--organism', type=str, default="homo_sapiens", help="Organism: mouse or human")
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


def plot_metrics_box(filtered_df, metric, ref_keys, group_col='study', outdir="outliers_kept", metric_label="Weighted F1", group_col_label="study"):
    levels = ref_keys
    for level in levels:
        level_df = filtered_df[filtered_df['key'] == level]
        if group_col in level_df.columns:
            #level_df[group_col] = level_df[group_col].str.strip().str.title()
            level_df = level_df.sort_values(by=group_col, key=lambda x: x.str.lower())
        if not level_df.empty:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_title(f"Level: {level}", fontsize=16, loc='left')
            sns.boxplot(data=level_df, y=group_col, x=metric, ax=ax, showfliers=False)
            means = level_df.groupby(group_col)[metric].mean().reset_index()
            mean_handle = None
            for i, row in means.iterrows():
                sc = ax.scatter(row[metric], row[group_col], color='black', marker='o', s=60, zorder=3)
                if mean_handle is None:
                    mean_handle = sc
            #if mean_handle is not None:
            #    ax.legend([mean_handle], ['Mean'], loc='upper left', bbox_to_anchor=(1.05, 1))
            ax.set_ylabel('', fontsize=MEDIUM_SIZE)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_xlim(-0.05, 1.05)
            ax.set_xticks(np.arange(0, 1.01, 0.2))
          #  metric_label = metric.replace('_', ' ').capitalize()
           # group_col_label = group_col.replace('_', ' ').capitalize()
            ax.set_xlabel(f'Agreement with author labels\n({metric_label})', fontsize=MEDIUM_SIZE)
            ax.legend().set_visible(False)
            ax.tick_params(axis='both', labelsize=SMALL_SIZE)
            plt.suptitle(f'{metric_label} grouped by {group_col_label}', fontsize=MEDIUM_SIZE)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{metric}_per_{group_col}_box_{level}.png"))
            plt.close()



def plot_metrics_strip(filtered_df, metric, ref_keys, group_col='study', outdir="outliers_kept", hue_color='label', metric_label="F1 Score", group_col_label="study"):
    levels = ref_keys
    for level in levels:
        level_df = filtered_df[filtered_df['key'] == level]
        if group_col in level_df.columns:
            #level_df[group_col] = level_df[group_col].str.strip().str.title()
            level_df = level_df.sort_values(by=group_col, key=lambda x: x.str.lower())
        if not level_df.empty:
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.set_title(f"Level: {level}", fontsize=16, loc='left')
            # Only stripplot for every point, colored by study, with jitter for visibility
            strip = sns.stripplot(data=level_df, y=group_col, x=metric, hue=hue_color, dodge=False, palette='tab10', size=8, alpha=0.8, jitter=True, ax=ax)
            ax.set_ylabel('', fontsize=MEDIUM_SIZE)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            # Add legend for hue (study)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=2, fontsize=SMALL_SIZE)
            ax.set_xlim(-0.05, 1.05)
            ax.set_xticks(np.arange(0, 1.01, 0.2))
          #  metric_label = metric.replace('_', ' ').capitalize()
          #  group_col_label = group_col.replace('_', ' ').capitalize()
            ax.set_xlabel(f'Agreement with author labels\n({metric_label})', fontsize=MEDIUM_SIZE)
          #  ax.legend().set_visible(False)
            ax.tick_params(axis='both', labelsize=SMALL_SIZE)
            plt.suptitle(f'{metric_label} grouped by {group_col_label}', fontsize=MEDIUM_SIZE)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{metric}_per_{group_col}_strip_{level}.png"))
            plt.close()

def smart_capitalize(s):
    s = s.strip()
    return s.capitalize() if s.islower() else s

        # Map PMIDs to study names
def map_study_to_pmid(study):
    return human_pmid_mapping.get(study, study)

def main():

    args = parse_arguments()
    weighted_df = pd.read_csv(args.weighted_metrics, sep='\t', low_memory=False)
    label_df = pd.read_csv(args.label_metrics, sep='\t', low_memory=False)
    ref_keys = args.ref_keys
    # Determine output directory based on outlier removal, reference, and cutoff
    ref_str = str(args.reference).replace(' ', '_')
    cutoff_str = f"cutoff_{args.cutoff}"
    if args.remove_outliers and len(args.remove_outliers) > 0:
        outdir = f"outliers_removed/{ref_str}/{cutoff_str}"
    else:
        outdir = f"outliers_kept/{ref_str}/{cutoff_str}"
    os.makedirs(outdir, exist_ok=True)
    # Remove outlier datasets if not None
    if args.remove_outliers:
        weighted_df = weighted_df[~weighted_df['study'].isin(args.remove_outliers)]
        label_df = label_df[~label_df['study'].isin(args.remove_outliers)]



    weighted_filtered = filter_metrics(weighted_df, args.subsample_ref, args.cutoff, args.reference, args.method)
    label_filtered = filter_metrics(label_df, args.subsample_ref, args.cutoff, args.reference, args.method)

    # Clean and capitalize study names for display

    # drop "Glia" and "VLMCeuron" - hack for now
    label_filtered = label_filtered[~label_filtered['label'].isin(["Glia", "VLMCeuron"])]
    
    # apply smart capitalization 
    label_filtered['study'] = label_filtered['study'].apply(smart_capitalize)
    weighted_filtered['study'] = weighted_filtered['study'].apply(smart_capitalize)
    #if args.organism == "homo_sapiens":
        #weighted_filtered['study'] = weighted_filtered['study'].apply(map_study_to_pmid)
        #label_filtered['study'] = label_filtered['study'].apply(map_study_to_pmid)


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
    
    # check if these columns exist in the dataframe
    available_metrics = [metric for metric in metrics_to_agg_weighted if metric in weighted_filtered.columns]
    metrics_to_agg_weighted = available_metrics
    
    metrics_to_agg_label = ['f1_score', 'precision', 'recall']
    
    
    
    # ------------------ grant summary plots ------------------

    # change this to a loop over metrics
    # plot weighted metrics
    for metric in metrics_to_agg_weighted:
        
        metric_label=metric.replace('_', ' ').capitalize()
        # replace f1 with F1
        metric_label = metric_label.replace('f1', 'F1')
        group_col_label='study'
        plot_metrics_box(weighted_filtered, metric=metric, ref_keys=ref_keys, group_col='study', outdir=outdir, metric_label=metric_label, group_col_label=group_col_label)
        plt.close()



    plot_metrics_box(label_filtered, metric="f1_score", ref_keys=ref_keys, group_col='label', outdir=outdir, metric_label="F1", group_col_label="label")
    plt.close()
    
    
    # Plot dots for label-level metrics, grouped by study, colored by label
  #  plot_metrics_box(label_filtered, metric="f1_score", ref_keys=ref_keys, group_col='study', outdir=outdir)
    plot_metrics_strip(label_filtered, metric="f1_score", ref_keys=ref_keys, group_col='study', outdir=outdir, hue_color='label', metric_label="F1", group_col_label="study")
    plt.close()
    
    
    # ------------------------ file summaries ------------------------
    
    # Summary of means and SDs for all metrics across all studies (weighted)
    weighted_summary = weighted_filtered[metrics_to_agg_weighted + ["key"]].groupby('key').agg(['mean', 'std']).T.reset_index()
    # set column names for only the first two columns
    
    weighted_summary.columns = ['metric', 'stat']  + weighted_summary.columns.tolist()[2:]
    weighted_summary.to_csv(f'{outdir}/sample_metrics_summary_overall.tsv', sep='\t', index=False)

    # Summary of means and SDs for all metrics across all cell types (label)
    # stratify by "key"
    label_summary = label_filtered[metrics_to_agg_label + ["key"]].groupby('key').agg(['mean', 'std']).T.reset_index()
    # set column names
    label_summary.columns = ['metric', 'stat'] + weighted_summary.columns.tolist()[2:]
    label_summary.to_csv(f'{outdir}/label_metrics_summary_overall.tsv', sep='\t', index=False)
    
    weighted_long = aggregate_metrics_long(
        weighted_filtered,
        groupby_col='study',
        metrics_to_agg=metrics_to_agg_weighted
    )
    weighted_long.to_csv(f"{outdir}/sample_metrics_stats_per_study.tsv", sep="\t", index=False)

    label_long = aggregate_metrics_long(
        label_filtered,
        groupby_col='label',
        metrics_to_agg=metrics_to_agg_label
        )
    label_long.to_csv(f"{outdir}/label_metrics_stats_per_label.tsv", sep="\t", index=False)
    
    
    
    
    
    
    #---------------------------- failed plots -----------------------------

    # plot_metrics_box(label_filtered, metric="accuracy", ref_keys=ref_keys, group_col='label')
  # plot_metrics_box(label_filtered, metric="precision", ref_keys=ref_keys, group_col='label')
  #  plot_metrics_box(label_filtered, metric="recall", ref_keys=ref_keys, group_col='label')

    # fails because precision is NA for some cell types due to lack of true positive predictions in that sample
    # this on a per-sample basis
    # computing overall precision and recall per cell type across samples addresses this
    # stored in another repo


if __name__ == "__main__":
    main()
